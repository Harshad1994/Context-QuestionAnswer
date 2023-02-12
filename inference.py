import openai
import os

from config import ft_discriminator,ft_qa


#uncomment the following like and provide the openai api key
# openai.api_key = $API_KEY


def question_answer_model(context,question):

    FINE_TUNED_MODEL=ft_qa

    PROMPT = context+"\n"+question+"\nAnswer:"

    response = openai.Completion.create(
        model=FINE_TUNED_MODEL,
        prompt=PROMPT, max_tokens=30, temperature=0, top_p=1, n=1, stop=['.','\n'])

    result=response.choices[0].text

    return result


def discriminator_model(context,question):

    FINE_TUNED_MODEL=ft_discriminator

    PROMPT = context+"\n"+question+"\n Related:"

    response = openai.Completion.create(
        model=FINE_TUNED_MODEL,
        prompt=PROMPT,
        max_tokens=1, temperature=0, top_p=1, n=1, logprobs=2)

    logprobs= response['choices'][0]['logprobs']['top_logprobs']

    logprobs=dict(logprobs[0])
    yes_logprobs=logprobs[' yes'] if ' yes' in logprobs else -100
    no_logprobs=logprobs[' no'] if ' no' in logprobs else -100

    if yes_logprobs < no_logprobs:
        return " no"
    else:
        return " yes"

def answer_question_conditionally(context,question):

    related=discriminator_model(context,question)

    if related == " no":
        return " No appropriate context found to answer the question based on the discriminator."
    
    return question_answer_model(context,question)

    

if __name__ == '__main__':
    context="The song was released as a digital download on 25 September 2015. It received mixed reviews from critics and fans, particularly in comparison to Adele's \"Skyfall\". The mixed reception to the song led to Shirley Bassey trending on Twitter on the day it was released. It became the first Bond theme to reach number one in the UK Singles Chart. The English band Radiohead also composed a song for the film, which went unused."
    question="Which English band also composed a song for the film?"
    answer=answer_question_conditionally(context,question)
    print(answer)