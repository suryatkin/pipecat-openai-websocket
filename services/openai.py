#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import wave
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.services.ai_services import STTService
from pipecat.utils.time import time_now_iso8601
from pipecat.transcriptions.language import Language

try:
    from openai import AsyncOpenAI
    from openai.types.audio import Transcription
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class OpenAISTTService(STTService):
    class InputParams(BaseModel):
        sample_rate: Optional[int] = 16000
        language: Optional[Language] = Language.EN
    
    def __init__(
        self,
        *,
        model: str = "whisper-1",
        api_key: str,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model: str = model
        self._client = AsyncOpenAI(api_key=api_key)
        self._settings = {
            "sample_rate": params.sample_rate,
            "language": params.language if params.language else Language.EN,
        }

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_ttfb_metrics()
            await self.push_frame(UserStartedSpeakingFrame())
            await self.push_frame(UserStoppedSpeakingFrame())

            # WAV parameters
            num_channels = 1       # 1 for mono, 2 for stereo
            sample_width = 2       # 2 bytes for 16-bit PCM
            sample_rate = self._settings["sample_rate"]

            content = io.BytesIO()
            ww = wave.open(content, "wb")
            ww.setsampwidth(sample_width)
            ww.setnchannels(num_channels)
            ww.setframerate(sample_rate)
            ww.writeframes(audio)
            ww.close()
            content.seek(0)

            response: Transcription = await self._client.audio.transcriptions.create(
                file=("audio.wav", content.read(), "audio/wav"),
                model=self._model,
                language='en'
            )

            await self.stop_ttfb_metrics()

            transcript = response.text.strip()

            if transcript:
                logger.debug(f"Transcription: [{transcript}]")
                await self.push_frame(
                    TranscriptionFrame(transcript, "", time_now_iso8601())
                )
                # yield TranscriptionFrame(transcript, "", time_now_iso8601())
            else:
                logger.warning("Received empty transcription from API")

        except Exception as e:
            logger.exception(f"Exception during transcription: {e}")
            yield ErrorFrame(f"Error during transcription: {str(e)}")