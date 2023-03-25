from twilio.rest import Client


def call_to_user(to_number):
    try:
        # Your Twilio account SID and auth token
        account_sid = 'AC75f288908ba8728dacaf1eb2718d5ef1'
        auth_token = '6bab6a26536ceb3eb213dc07a29e7a44'

        # Create a Twilio client with your account SID and auth token
        client = Client(account_sid, auth_token)

        # The phone number to call from (in E.164 format and must be a Twilio verified phone number)
        from_number = '+15074364355'

        # Make the phone call
        call = client.calls.create(
            to="+91"+str(to_number),
            from_=from_number,
            url='http://demo.twilio.com/docs/voice.xml'  # TwiML instructions for the call
            )

        # Print the call SID
        print(call.sid)
    except:
        call = "mobile number not verified"
        
    return call
