# ComfyUI-LTXVideo-AVSplit

Пак нод для ComfyUI с разделением/сшивкой AV-латентов LTX и расчетом разрешения под staged upscale.

В паке 4 ноды:
- `LTXVSetAudioVideoMaskByTimeSplit`
- `LTXVStitchAVLatentsWithTransitionMask`
- `LTXVPostBlendTransition`
- `TwoStageResolution`

Display-имена в ComfyUI формируются с префиксом `🅛🅣🅧`.

## Установка

1. Положите папку в `ComfyUI/custom_nodes/ComfyUI-LTXVideo-AVSplit`.
2. Установите зависимости:
   `pip install -r requirements.txt`
3. Перезапустите ComfyUI.

## Совместимость и ограничения

- AV-ноды работают только с моделями, где `model.model.diffusion_model` это `LTXAVModel`.
- Входные AV-латенты должны быть в формате `NestedTensor` в `latent["samples"]`.
- Если одновременно установлен оригинальный `ComfyUI-LTXVideo`, следите за конфликтами Node ID.

## Сводка нод

| Node ID | Category | Входы | Выходы | Назначение |
| --- | --- | --- | --- | --- |
| `LTXVSetAudioVideoMaskByTimeSplit` | `lightricks/LTXV` | `LATENT`, `MODEL`, `VAE`, `VAE` + тайминги/маски | `LATENT` | Создает/обновляет `noise_mask` с раздельными окнами по времени для видео и аудио |
| `LTXVStitchAVLatentsWithTransitionMask` | `lightricks/LTXV` | 2x `LATENT`, `MODEL` + режим сшивки/маски | `LATENT` | Сшивает два AV-латента, формирует transition и маски |
| `LTXVPostBlendTransition` | `lightricks/LTXV` | `LATENT` (denoised), `LATENT` (stitch), `MODEL` | `LATENT` | Пост-бленд исходных и сгенерированных латентов после равномерного денойза |
| `TwoStageResolution` | `video/resolution` | `width`, `height`, `spatial_upscaler` | `width`, `height`, `info` | Считает базовое разрешение под выбранный upscale-режим |

## Ноды подробно

### 1) `LTXVSetAudioVideoMaskByTimeSplit`

Назначение: построить `noise_mask` для AV-латента с отдельными временными окнами и slope для видео/аудио.

Обязательные входы:
- `av_latent` (`LATENT`): AV-латент.
- `model` (`MODEL`): модель LTX AV.
- `vae` (`VAE`): нужен для пересчета video time -> latent indices.
- `audio_vae` (`VAE`): нужен для пересчета audio time -> latent indices.
- `video_fps` (`FLOAT`): FPS декодированного видео.
- `video_start_time`, `video_end_time` (`FLOAT`): временное окно видео в секундах.
- `video_slope_len` (`INT`): длина плавного входа/выхода маски для видео.
- `mask_video` (`BOOLEAN`): включить видео-маску.
- `mask_init_value_video` (`FLOAT`): базовое значение маски видео до envelope.
- `audio_start_time`, `audio_end_time` (`FLOAT`): временное окно аудио в секундах.
- `audio_slope_len` (`INT`): длина slope для аудио.
- `mask_audio` (`BOOLEAN`): включить аудио-маску.
- `mask_init_value_audio` (`FLOAT`): базовое значение маски аудио.

Опциональные входы:
- `spatial_mask` (`MASK`): 2D `(H,W)` или 3D `(T,H,W)` маска для видео; масштабируется к latent-size и смешивается с temporal envelope.
- `merge_existing_video_mask` (`BOOLEAN`): если у входного латента уже есть `noise_mask`, домножает новую видео-маску на существующую per-frame scalar.

Выход:
- `av_latent` (`LATENT`): копия входа с новым `noise_mask`.

Ключевое поведение:
- Маска строится как envelope в диапазоне `[0..1]` с плавными краями.
- Индексы времени автоматически clamp к реальной длине латента.
- Для `spatial_mask` используется blend, а не жесткая замена.

### 2) `LTXVStitchAVLatentsWithTransitionMask`

Назначение: сшить два AV-латента (`av_latent_1`, `av_latent_2`) и сформировать transition + маски.

Основные параметры:
- `stitch_mode`:
  - `bridge`: вставляет отдельный bridge-сегмент между клипами.
  - `overlap_linear_video`: линейно смешивает overlap хвоста 1-го и головы 2-го клипа (только для видео).
- `bridge_latent_frames`: длина bridge в latent frames.
- `overlap_latent_frames`: длина overlap для `overlap_linear_video`.
- `bridge_init_mode`:
  - `lerp`: линейная интерполяция между концами.
  - `noise`: случайный Gaussian bridge.
  - `zeros`: нулевой bridge.
- `video_pre_frames`, `video_post_frames`, `video_slope_len`: управление окном и краями видео-маски вокруг transition.
- `mask_video`, `video_mask_init_value`: включение и база видео-маски.
- `audio_start_time`, `audio_end_time`, `audio_slope_len`, `mask_audio`, `audio_mask_init_value`: параметры аудио-маски.
- `post_blend` (`BOOLEAN`):
  - `False`: обычный режим, пишет `noise_mask` в выходной latent.
  - `True`: не ставит `noise_mask`, а сохраняет служебные поля для `LTXVPostBlendTransition`.
- Опционально `vae`, `audio_vae`: точный temporal stride/аудио-rate; если не подключены, берутся дефолты (`8` и `~25 Hz`).

Выход:
- `av_latent` (`LATENT`): сшитый латент.

Ключевое поведение:
- Аудио всегда сшивается через bridge (даже в `overlap_linear_video`).
- Размер transition-области учитывается при построении масок.
- В `post_blend=True` в latent сохраняются:
  - `_postblend_video_original`
  - `_postblend_audio_original`
  - `_postblend_video_blend`
  - `_postblend_audio_blend`

### 3) `LTXVPostBlendTransition`

Назначение: пост-бленд после sampling, когда stitch-нода работала с `post_blend=True`.

Входы:
- `denoised_av_latent` (`LATENT`): результат сэмплера.
- `stitch_av_latent` (`LATENT`): выход `LTXVStitchAVLatentsWithTransitionMask` из той же ветки с `post_blend=True`.
- `model` (`MODEL`): модель LTX AV.

Выход:
- `av_latent` (`LATENT`): итоговый латент после смешивания.

Формула смешивания:
- `output = original * (1 - blend) + denoised * blend`

Ключевое поведение:
- Проверяет наличие служебных `_postblend_*` полей.
- Удаляет `noise_mask` из результата (если он был), чтобы вернуть чистый blended latent.

### 4) `TwoStageResolution`

Назначение: вычислить базовое разрешение, согласованное с шагами латента и выбранным upscale-режимом.

Входы:
- `width` (`INT`)
- `height` (`INT`)
- `spatial_upscaler` (`none`, `1.5`, `2`, `3KS (2x+2x)`)

Выходы:
- `width` (`INT`): рассчитанный base width.
- `height` (`INT`): рассчитанный base height.
- `info` (`STRING`): человекочитаемая сводка (`BASE`, `MID`, `FACT`).

Режимы:
- `none`: округление до шага `32`.
- `1.5` или `2`: подбор шага, кратного и `32`, и коэффициенту upscale.
- `3KS (2x+2x)`: расчет цепочки `BASE -> MID -> FACT`, где `FACT` кратен `128`.

## Минимальная схема использования Stitch + PostBlend

1. Подайте два латента в `LTXVStitchAVLatentsWithTransitionMask`.
2. Включите `post_blend=True`.
3. Выход stitch-ноды отправьте в sampler.
4. Результат sampler (`denoised_av_latent`) и исходный выход stitch-ноды (`stitch_av_latent`) подайте в `LTXVPostBlendTransition`.
5. Используйте выход `LTXVPostBlendTransition` как финальный AV-латент.

## Зависимости

- `numpy` (см. `requirements.txt`)
