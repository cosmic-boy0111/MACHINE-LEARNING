
import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import os
import smtplib

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        print('Good Morning!')
        speak("Good Morning!")
    elif hour>=12 and hour<18:
        print('Good Afternoon!')
        speak("Good Afternoon!")
    else:
        print("Good Evening!")
        speak("Good Evening!")

    speak(f"I am Friday Sir, and it is {datetime.datetime.now().strftime('%H:%M')}. Please tell me how may i help you")


def takeCommand(previous_command):
    # it takes microphone input from user and return string as output

    r = sr.Recognizer() # it helps to recognize the voice of audio input
    with sr.Microphone() as source:
        # pip install pipwin
        # pipwin install pyaudio
        print("Listening....")
        r.pause_threshold = 1
        # r.energy_threshold = 301
        audio = r.listen(source)

    try:
        print('Recognizing....')
        query = r.recognize_google(audio,language='en-in')
        print(f"User said : {query}\n")
    except:
        print('Say that again please.....')
        return 'None'

    if 'stop' in query.lower():
        speak('Thank you sir, we will meet soon..')
        exit()

    if 'repeat again' in query.lower():
        return previous_command
    return query

def sendEmail(to,cointaint):
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.ehlo()
    server.starttls()
    server.login('gauravbhagat2040@gmail.com','1234@Gaurav')
    server.sendmail('gauravbhagat2040@gmail.com',to,cointaint)
    server.close()


if __name__=="__main__":
    wishMe()
    previous_command = ''
    while True:
        query = takeCommand(previous_command).lower()
        previous_command = query
        #logic for executing task we have to perform
        if 'wikipedia' in query:
            try:
                speak('Searching Wikipedia....')
                query = query.replace('wikipedia','')
                results = wikipedia.summary(query, sentences=1)
                speak('According to wikipedia.')
                print(results)
                speak(results)
            except:
                speak('Result not found, try again!')
        
        elif 'open youtube' in query:
            webbrowser.open('youtube.com')
        
        elif 'open google' in query:
            webbrowser.open('google.com')
        
        elif 'open stack overflow' in query:
            webbrowser.open('stackoverflow.com')

        elif 'play music' in query:
            music_dic = 'C:\\Users\\Gaurav\\Music'
            songs = os.listdir(music_dic)
            # print(songs)
            os.startfile(os.path.join(music_dic,songs[1]))

        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime('%H:%M:%S')
            speak(f'Sir, the time is {strTime}')

        elif 'open code' in query:
            code_path = "C:\\Users\\Gaurav\\AppData\\Local\\Programs\\Microsoft VS Code\\Code"
            os.startfile(code_path)

        elif 'send email' in query:
            #https://myaccount.google.com/lesssecureapps?pli=1&rapt=AEjHL4PBcMF4n72CRq0ZGT3e7CsfLXINor0Y6092-20a3txCY0OmRme5CnPmfvbMW9UuUnTgw3_6jSW2WJayRlSnMCyftP_QuQ
            try:
                speak('What should I say :')
                cointaint = takeCommand(previous_command)
                to = 'khatalegayatri20@gmail.com'
                sendEmail(to,cointaint)
                speak('Email has been send')
            except Exception as e:
                print(e)
                print('Sorry Sir, I am not able to send this email at the moment !')
        
        elif 'who are you' in query:
            speak('I am Friday, an AI, how may i help you')

        else:
            speak('something wents wrong. Please try again!')


        

