# main.py - This file contains the main logic for the Discord bot.
import os
import discord
from google.cloud import speech, texttospeech, aiplatform
from discord.ext import commands
import asyncio
import pyaudio
import threading
import numpy as np
from precise_runner import PreciseEngine, PreciseRunner

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/workspaces/Discord-Assistant/credentials.json'

# Initialize Google Cloud clients
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
aiplatform.init()

# Set up Discord bot with necessary intents for interacting with users and voice channels
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.voice_states = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Audio recording setup
FORMAT = pyaudio.paInt16  # Audio format for recording
CHANNELS = 1  # Number of audio channels
RATE = 16000  # Sampling rate
CHUNK = 1024  # Size of each audio chunk
audio = pyaudio.PyAudio()  # PyAudio instance for recording

# Path to Precise wake word model and engine
ENGINE_PATH = '/workspaces/Discord-Assistant/precise-engine'
MODEL_PATH = '/workspaces/Discord-Assistant/hey-mycroft.pb'

# Callback function for when the wake word is detected
def on_wake():
    print("Wake word detected!")
    global wake_word_detected
    wake_word_detected = True

# Initialize Precise engine and runner for wake word detection
engine = PreciseEngine(ENGINE_PATH, MODEL_PATH)
runner = PreciseRunner(engine, on_activation=on_wake)
runner.start()

wake_word_detected = False  # Global variable to track if wake word is detected

@bot.event
async def on_ready():
    # Event triggered when the bot is ready
    print(f'Logged in as {bot.user}')

@bot.command(name='join')
async def join(ctx):
    """Command for the bot to join the user's voice channel."""
    if ctx.author.voice:
        # If the author is in a voice channel, join it
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"Joined {channel}")
    else:
        # If the author is not in a voice channel
        await ctx.send("You need to be in a voice channel for me to join!")

@bot.command(name='leave')
async def leave(ctx):
    """Command for the bot to leave the voice channel."""
    if ctx.voice_client:
        # If the bot is in a voice channel, disconnect
        await ctx.guild.voice_client.disconnect()
        await ctx.send("Disconnected from the voice channel.")
    else:
        # If the bot is not in a voice channel
        await ctx.send("I'm not in a voice channel.")

@bot.command(name='listen')
async def listen(ctx, user: discord.Member):
    """Command to start listening for a specific user's audio input."""
    if not ctx.voice_client:
        # Ensure the bot is in a voice channel before listening
        await ctx.send("I need to be in a voice channel to listen. Use !join to invite me.")
        return

    await ctx.send(f"Listening for {user.display_name}'s voice...")
    voice_client = ctx.voice_client
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []  # List to store audio frames
    global wake_word_detected
    wake_word_detected = False

    while True:
        # Read audio chunk from the stream
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, np.int16)

        # Check for wake word using Precise runner
        if not wake_word_detected:
            runner.engine().detect(audio_data)
            if wake_word_detected:
                await ctx.send("Wake word detected. Start speaking...")
        else:
            frames.append(data)  # Start recording after wake word is detected
            silent_chunks = 0
            silence_threshold = 200  # Threshold to determine silence in the audio
            max_silent_chunks = int(RATE / CHUNK * 1)  # Stop after 1 second of silence

            # Continue listening until silence is detected
            while True:
                data = stream.read(CHUNK)
                frames.append(data)

                # Check if the audio is silent
                if max(np.frombuffer(data, np.int16)) < silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                # Stop if silence is detected for a certain duration
                if silent_chunks > max_silent_chunks:
                    break

            break  # Exit the loop after recording is done

    # Stop the audio stream
    stream.stop_stream()
    stream.close()

    # Send the recorded audio to Google Speech-to-Text for transcription
    audio_content = b''.join(frames)
    audio_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US"
    )
    audio_data = speech.RecognitionAudio(content=audio_content)

    response = speech_client.recognize(config=audio_config, audio=audio_data)

    if response.results:
        # If transcription is successful, get the transcript
        transcript = response.results[0].alternatives[0].transcript
        await ctx.send(f"{user.display_name} said: {transcript}")

        # Use Google Vertex AI to generate a response based on the transcript
        vertex_ai_model = aiplatform.TextGenerationModel.from_pretrained("text-bison")
        vertex_response = vertex_ai_model.predict(transcript)
        generated_text = vertex_response.predictions[0]['content']

        # Generate a response using Google Text-to-Speech
        synthesis_input = texttospeech.SynthesisInput(text=generated_text)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        tts_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        # Play the synthesized speech directly without saving
        voice_client.play(discord.FFmpegPCMAudio(source=tts_response.audio_content), after=lambda e: print('Finished playing response'))
    else:
        # If transcription fails
        await ctx.send("Sorry, I couldn't understand what you said.")

# Run the bot with the provided token
bot.run('YOUR_DISCORD_BOT_TOKEN')