from app.ai.ner_model import extract_symptoms_from_text
from app.ai.predict import predict_disease

def print_colored(text, color_code='32'):
    return f"\033[{color_code}m{text}\033[0m"

while True:
    user_input = input(print_colored('I am a bot designed to help you identify your disease based on the symptoms you provide. Start by explaining your symptoms:\n'))
    symptoms = extract_symptoms_from_text(user_input)

    print(print_colored(f"Entered symptoms: {symptoms}", '33'))
    print(print_colored(f"You may have: ", '34'), print_colored(predict_disease(','.join(symptoms)), '35'))
