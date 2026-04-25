# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for HunyuanImage3: AR → Diffusion transition.

In IT2I (image editing) mode:
  - Stage 0 (AR) receives (image + edit instruction), generates CoT/latent tokens
  - Stage 1 (DiT) receives the AR output + original image, denoises → edited image

The ar2diffusion function bridges these two stages, following the same
signature pattern as glm_image.ar2diffusion.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def ar2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Process AR stage outputs to create Diffusion stage inputs.

    Args:
        stage_list: List of stage clients (set by orchestrator).
        engine_input_source: List of source stage IDs (from YAML).
        prompt: Original user prompt (may contain multimodal data).
        requires_multimodal_data: Whether to forward multimodal data.

    Returns:
        List of dicts, each consumable by the HunyuanImage3 diffusion pipeline.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid source stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    ar_outputs = stage_list[source_stage_id].engine_outputs
    diffusion_inputs = []

    # Normalize prompt to list
    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_output in enumerate(ar_outputs):
        output = ar_output.outputs[0]
        generated_token_ids = output.token_ids
        generated_text = getattr(output, "text", "") or ""

        # When the AR stage is configured with `detokenize: false` (the default
        # for the HunyuanImage-3 IT2I configs, which avoids the cost of streaming
        # text during generation), ``output.text`` is empty even though tokens
        # were produced.  The DiT needs the decoded CoT text to rebuild the
        # joint sequence via ``apply_chat_template``, so decode the tokens here
        # on the fly as a fallback.  Without this fallback the DiT receives an
        # empty ``ar_generated_text`` and silently drops the image conditioning
        # — this is the "text length=0 / model ignores the input image" symptom
        # reported in https://github.com/vllm-project/vllm-omni/pull/2590.
        if not generated_text and generated_token_ids:
            tokenizer = _resolve_ar_tokenizer(stage_list[source_stage_id])
            if tokenizer is not None:
                try:
                    generated_text = tokenizer.decode(list(generated_token_ids), skip_special_tokens=False)
                except Exception as exc:
                    logger.warning(
                        "[ar2diffusion] Failed to decode AR tokens for request %d: %s",
                        i,
                        exc,
                    )

        # Get original prompt info
        original_prompt = prompt[i] if i < len(prompt) else {}
        if isinstance(original_prompt, dict):
            pass
        elif hasattr(original_prompt, "_asdict"):
            original_prompt = original_prompt._asdict()
        elif hasattr(original_prompt, "__dict__"):
            original_prompt = vars(original_prompt)
        else:
            original_prompt = {}

        height = original_prompt.get("height", 1024)
        width = original_prompt.get("width", 1024)
        # Prefer clean user_prompt over the full baked AR string so the DiT
        # can reconstruct the sequence with the same template as at training time.
        # Fall back to "prompt" for callers that do not populate "user_prompt".
        text_prompt = original_prompt.get("user_prompt") or original_prompt.get("prompt", "")
        use_system_prompt = original_prompt.get("use_system_prompt")

        logger.info(
            "[ar2diffusion] Request %d: AR generated %d tokens, text length=%d, target size=%dx%d",
            i,
            len(generated_token_ids),
            len(generated_text),
            height,
            width,
        )

        # If the AR stage used a pretrain-format input (KV-reuse path) the
        # trigger tag (e.g. "<think>") was NOT part of the AR prefill and is
        # provided separately via ``original_prompt["trigger_tag"]``.  Concatenate
        # it here (mirroring the official HunyuanImage-3 reference:
        # https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/hunyuan_image_3/modeling_hunyuan_image_3.py#L3355)
        # so the DiT just sees a single ``ar_generated_text`` field without
        # needing to know about the trigger tag.  Guard against double-prepend
        # when the AR model has already emitted the trigger as its first token.
        trigger_tag = original_prompt.get("trigger_tag")
        if trigger_tag and generated_text and not generated_text.startswith(trigger_tag):
            generated_text = trigger_tag + generated_text

        token_tensor = torch.tensor(generated_token_ids, dtype=torch.long)

        diffusion_input: dict[str, Any] = {
            "prompt": text_prompt,
            "height": height,
            "width": width,
            "extra": {
                "ar_token_ids": token_tensor,
                "ar_generated_text": generated_text,
            },
        }

        # Forward use_system_prompt so the DiT can build the same system prefix
        if use_system_prompt is not None:
            diffusion_input["use_system_prompt"] = use_system_prompt

        # Forward multimodal data (original image for IT2I conditioning)
        mm_data = original_prompt.get("multi_modal_data")
        if mm_data:
            prompt_images = mm_data.get("image")
            if prompt_images is None:
                prompt_images = mm_data.get("images")
            if prompt_images is not None:
                diffusion_input["pil_image"] = prompt_images
                diffusion_input["multi_modal_data"] = {"image": prompt_images}

        # Forward multimodal output from AR (if any)
        if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
            mm_output = ar_output.multimodal_output
            if isinstance(mm_output, dict):
                diffusion_input["extra"]["ar_multimodal_output"] = mm_output

        # Forward sampling params
        for key in ["seed", "num_inference_steps", "guidance_scale", "negative_prompt"]:
            if key in original_prompt:
                diffusion_input[key] = original_prompt[key]

        diffusion_inputs.append(diffusion_input)

    return diffusion_inputs


logger = logging.getLogger(__name__)


_AR_TOKENIZER_CACHE: dict[str, Any] = {}


def _resolve_ar_tokenizer(stage_client: Any) -> Any:
    """Best-effort resolution of the AR stage's tokenizer.

    The stage client exposes the resolved ``vllm_config``; we use its
    ``model_config.model`` path to load an ``AutoTokenizer`` lazily and cache
    the result so we don't pay the init cost on every request.
    """
    model_path = None
    vllm_config = getattr(stage_client, "vllm_config", None)
    if vllm_config is not None:
        model_cfg = getattr(vllm_config, "model_config", None)
        if model_cfg is not None:
            model_path = getattr(model_cfg, "tokenizer", None) or getattr(model_cfg, "model", None)
    if model_path is None:
        model_path = getattr(stage_client, "model", None) or getattr(stage_client, "model_name", None)
    if not model_path:
        return None
    if model_path in _AR_TOKENIZER_CACHE:
        return _AR_TOKENIZER_CACHE[model_path]
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning("[ar2diffusion] Could not load tokenizer from %r: %s", model_path, exc)
        _AR_TOKENIZER_CACHE[model_path] = None
        return None
    _AR_TOKENIZER_CACHE[model_path] = tokenizer
    return tokenizer
