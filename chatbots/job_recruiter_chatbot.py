# Install the following requirements:
# DialogFlow 0.5.1
# google-api-core 1.4.1
import dialogflow
from google.api_core.exceptions import InvalidArgument
import os

"""
    This chatbot uses Google's Dialogflow API that is configured on google cloud

    set PROJECT_ID=<google cloud project id>
    set GOOGLE_APPLICATION_CREDENTIALS=<google cloud service account credentials as a json file>
    set SESSION_ID=<google cloud service account>        
    python job_recruiter_chatbot.py
    
    Sample conversation with this bot will be something like
    
            Query  text: hi there
            Fulfillment text: How can I help you with job search today?
            ############################################################
            Query  text: Have you made a decision about the position I applied to?
            Fulfillment text: Let me look into it. Can you give me the position id and position description please?
            ############################################################
            Query  text: It is 999999 for a Software developer position
            Fulfillment text: We will let you know in a couple of days.
            ############################################################
            Query  text: Thanks
            Fulfillment text: Is there anything else I can help you with?
            ############################################################
            Query  text: No
            Fulfillment text: Have a wonderful rest of your day.
            ############################################################
"""
class JobHireChatBot():

    def __init__(self):
        self.PROJECT_ID = os.getenv('PROJECT_ID')
        self.LANGUAGE_CODE = 'en-US'
        self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.SESSION_ID = os.getenv('SESSION_ID')
        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(self.PROJECT_ID, self.SESSION_ID)


    def chatNow(self):
        self.askMyChatBot("hi there")
        self.askMyChatBot("Have you made a decision about the position I applied to?")
        self.askMyChatBot("It is 999999 for a Software developer position")
        self.askMyChatBot("Thanks")
        self.askMyChatBot("No")

    def askMyChatBot(self, user_input):
        try:
            text_input = dialogflow.types.TextInput(text=user_input, language_code=self.LANGUAGE_CODE)
            query_input = dialogflow.types.QueryInput(text=text_input)
            response = self.session_client.detect_intent(session=self.session, query_input=query_input)
            self.printResponse( response)
        except InvalidArgument:
            raise

    def printResponse(self, response):
        print("Query  text:", response.query_result.query_text)
        # print("Detected intent:", response.query_result.intent.display_name)
        # print("Detected intent confidence:", response.query_result.intent_detection_confidence)
        print("Fulfillment text:", response.query_result.fulfillment_text)
        print("############################################################")


if __name__ == "__main__":
    myJobHireBot = JobHireChatBot()
    myJobHireBot.chatNow()
