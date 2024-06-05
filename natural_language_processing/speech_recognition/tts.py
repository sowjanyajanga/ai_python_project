import pyttsx4

# I needed to add "import objc" to nsss.py
# https://stackoverflow.com/questions/77256187/pyttsx3-nsss-py-gives-an-weird-obj-not-defined-error
engine = pyttsx4.init()
rate = engine.getProperty('rate')

engine.setProperty('rate', rate-50)
voices = engine.getProperty('voices')
engine.setProperty('voice', 'TTS_MS_EN-US_ZIRA_11.0')

engine.say("You are reading the Python Machine Learning Cookbook")
engine.say("I hope you like it.")

engine.runAndWait()

