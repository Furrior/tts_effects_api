import asyncio
import base64
from enum import Enum
import time
import logging

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import ujson
import uvicorn

class VoiceRequest(BaseModel):
    api_token: str
    text: str
    sample_rate: int = 24000
    ssml: bool = False
    speaker: str = 'xenia'
    remote_id: str = 'empty'
    lang: str = 'ru'
    put_accent: bool = True
    put_yo: bool = True
    symbol_durs: list = []
    format: str = 'ogg'
    word_ts: bool = False

    sfx: list[str] | None = None


class FFmpegFlters(Enum):
    radio = ["-filter:a", "highpass=f=1000, lowpass=f=3000, acrusher=1:1:50:0:log"]

    robot = ["-filter:a",
             "afftfilt=real='hypot(re,im)*sin(0)':imag='hypot(re,im)*cos(0)':win_size=512:overlap=0.7, deesser=i=0.1, volume=volume=1.3"]
    megaphone = ["-filter:a", "highpass=f=500, lowpass=f=4000, volume=volume=10, acrusher=1:1:45:0:log"]
    
    # Archive/Shitspawn
    tg_robot = ["-i", "./SynthImpulse.wav", "-i", "./RoomImpulse.wav", "-filter_complex",
                "[0] aresample=44100 [re_1]; [re_1] apad=pad_dur=2 [in_1]; [in_1] asplit=2 [in_1_1] [in_1_2]; [in_1_1] [1] afir=dry=10:wet=10 [reverb_1]; [in_1_2] [reverb_1] amix=inputs=2:weights=8 1 [mix_1]; [mix_1] asplit=2 [mix_1_1] [mix_1_2]; [mix_1_1] [2] afir=dry=1:wet=1 [reverb_2]; [mix_1_2] [reverb_2] amix=inputs=2:weights=10 1 [mix_2]; [mix_2] equalizer=f=7710:t=q:w=0.6:g=-6,equalizer=f=33:t=q:w=0.44:g=-10 [out]; [out] alimiter=level_in=1:level_out=1:limit=0.5:attack=5:release=20:level=disabled"]
    robot_aziz = ["-filter:a",
             "afftfilt=real='hypot(re,im)*sin(0)':imag='hypot(re,im)*cos(0)':win_size=1024:overlap=0.5, deesser=i=0.4, volume=volume=1.5"]

async def do_silero_request(data: dict) -> bytes:
    async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:
        async with session.post(settings.api_url, json=data) as resp:
            if resp.status != 200:
                raise HTTPException(resp.status)
            return await resp.read()


async def run_async_subprocess(command: list[str], input_data: bytes = None) -> tuple[bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate(input=input_data)
    return stdout, stderr


async def apply_ffmpeg_filter(audio: bytes, ffmpeg_filter: FFmpegFlters) -> tuple[bytes, bytes]:
    ffmpeg_command = ["ffmpeg", "-f", "ogg", "-i", "pipe:0", *ffmpeg_filter.value, "-c:a", "libvorbis", "-b:a",
                      "64k",
                      "-f", "ogg", "pipe:1"]

    stdout, stderr = await run_async_subprocess(ffmpeg_command, audio)
    # ffmpeg_metadata_output = stderr.decode()

    return stdout, stderr

async def process_sfxs(audio: bytes, sfxs: list) -> dict[str, bytes]:
    sfxs = sorted(sfxs)
    resulting_audios = {}
    for sfx in sfxs:
        stdout = audio
        try:
            ffmpeg_filter = FFmpegFlters[sfx]
        except KeyError:
            print(f"Got wrong filter - '{sfx}'. Ignoring...")
            continue
        stdout, _ = await apply_ffmpeg_filter(stdout, ffmpeg_filter)
        encoded_result = base64.b64encode(stdout)
        resulting_audios["_".join(sfxs)] = encoded_result
    return resulting_audios


class Settings(BaseSettings):
    api_url: str = "https://api-tts.silero.ai/voice"
    port: int = 10000

    class Config:
        env_file = ".env"


settings = Settings()
app = FastAPI()


@app.post("/voice")
async def translate_voice(request: VoiceRequest):
    body = request.model_dump()
    silero_response = await do_silero_request(body)
    response = ujson.loads(silero_response)
    encoded_audio = response['results'][0]['audio']
    audio = base64.b64decode(encoded_audio)
    resulting_audios = {"pure": encoded_audio}
    start_processing = time.time()

    if request.sfx:
        processed_audios = await process_sfxs(audio, request.sfx)
        resulting_audios.update(processed_audios)

    processing_time = time.time() - start_processing

    response['results'][0]['audio'] = resulting_audios
    response['timings']['220_sfx_time'] = processing_time
    return response


if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=settings.port, reload=True)
