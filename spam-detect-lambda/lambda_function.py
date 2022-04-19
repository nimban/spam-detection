import json
import email
import boto3
import string
import os
from sms_spam_classifier_utilities import one_hot_encode, vectorize_sequences

vocabulary_length = 9013


def lambda_handler(event, context):
    print("New Test ")
    print(event)
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    fileObj = boto3.client("s3").get_object(Bucket=bucket, Key=key)

    msg = email.message_from_bytes(fileObj['Body'].read())
    subject = msg['Subject']
    sent_datetime = msg["Date"]
    sender_email = msg['From']
    if '<' in sender_email:
        start = sender_email.find("<") + 1
        end = sender_email.find(">")
        sender_email = sender_email[start:end]
    sender_email = "niranjan119n@gmail.com"

    body = []
    if msg.is_multipart():
        for part in msg.get_payload():
            body.append(part.get_payload())
    else:
        body.append(msg.get_payload())
    body = body[0].replace('\n', '').replace('\r', '')

    body_list = []
    body_list.append(body)
    print(body_list)

    one_hot_test_messages = one_hot_encode(body_list, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    encoded_test_messages = encoded_test_messages.tolist()
    print("\n\nENCODED")
    print(encoded_test_messages)

    ENDPOINT_NAME = "sms-spam-classifier-mxnet-2022-04-15-04-38-23-956"

    runtime = boto3.client('runtime.sagemaker')
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       Body=json.dumps(encoded_test_messages))

    sm_response = json.loads(response["Body"].read())
    prediction = sm_response['predicted_label'][0][0]
    confidence = sm_response['predicted_probability'][0][0]
    if prediction == 1:
        confidence_per = confidence * 100
    else:
        confidence_per = (1 - confidence) * 100
    prediction_text = 'spam' if prediction == 1 else 'non-spam'
    create_email_string = '''We received your email sent at {} with the subject {}.

    Here is  the email body:
        {}

    The email was categorized as {} with a {}% confidence.'''.format(sent_datetime, subject, body, prediction_text,
                                                                     confidence_per)
    print(create_email_string)
    ses_client = boto3.client("ses")
    CHARSET = "UTF-8"

    response = ses_client.send_email(
        Destination={
            "ToAddresses": [
                sender_email
            ],
        },
        Message={
            "Body": {
                "Text": {
                    "Charset": CHARSET,
                    "Data": create_email_string,
                }
            },
            "Subject": {
                "Charset": CHARSET,
                "Data": "Spam Report",
            },
        },
        Source="niranjan119n@gmail.com",
    )
    print(response)
    return

