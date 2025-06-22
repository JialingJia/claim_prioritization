import numpy as np
import streamlit as st
import os
from openai import OpenAI
import json

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

class Template:
    def __init__ (self, facet, input, description, example):
        self.facet = facet
        self.input = input
        self.description = description
        self.example = example

    def des_context(self):
        context = """\n###\nContext:""" + self.description
        return context

    def ex_context(self):
        context = ""
        for idx, example in enumerate(self.example):
            if example[1] == 1:
                context_example = """\n###\nInput: """ + example[0] + """\n###\nOutput: yes"""
            if example[1] == 0:
                context_example = """\n###\nInput: """ + example[0] + """\n###\nOutput: no"""
            context = context_example
        return context

    def prompt(self):
        intro = """\n###\nInput:""" + self.input
        user_defined_instruct = """\n###\nIdentify whether the input claim is """ + f"""{self.facet}""" + """ and output yes or no."""
        output = """\n###\nOutput:"""
        if self.description and self.example:
            context = self.des_context() + self.ex_context()
        elif self.description:
            context = self.des_context()
        elif self.example:
            context = self.ex_context()
        else:
            context = ""

        prompt = user_defined_instruct + context + intro + output
        return prompt 
    
class GPT:
    def __init__(self, prompt):
        self.prompt = prompt

    def generate_1(self):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds with only 'yes' or 'no'."},
                {"role": "user", "content": self.prompt}
            ],
            max_tokens=10,
            temperature=0,
            logprobs=True,
            top_logprobs=5
        )
        
        # Extract token logprobs from the response
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        
        yes_prob = 0
        no_prob = 0
        
        # Search for 'yes' and 'no' tokens in the logprobs
        for item in logprobs:
            if item.token.strip().lower() == 'yes':
                yes_prob = np.exp(item.logprob)
            elif item.token.strip().lower() == 'no':
                no_prob = np.exp(item.logprob)
        
        # If neither 'yes' nor 'no' is in top tokens, check completion text
        if yes_prob == 0 and no_prob == 0:
            completion_text = response.choices[0].message.content.strip().lower()
            if completion_text == 'yes':
                yes_prob = 0.99
                no_prob = 0.01
            elif completion_text == 'no':
                yes_prob = 0.01
                no_prob = 0.99
            else:
                # Fallback if neither yes nor no is detected
                yes_prob = 0.5
                no_prob = 0.5
        
        yes_prob_normalized = round(yes_prob/(no_prob+yes_prob), 3)
        
        return [1, yes_prob_normalized]
        
    def generate_2(self):
        # This is now just an alias for generate_1 to maintain compatibility
        return self.generate_1()