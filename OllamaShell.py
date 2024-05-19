import torch
import ollama
import os
from openai import OpenAI
import argparse
import json
import subprocess

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def save_response_to_file(response, filename):
    with open(filename, 'a') as file:
        file.write(response + '\n\n')

def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    filtered_lines = []
    for line in lines:
        # Check if the line starts with a number followed by a period or a closing parenthesis
        if line.strip().startswith(tuple(str(i) for i in range(10))) and ('.' in line or ')' in line):
            # Check if the line contains a closing parenthesis
            if ')' in line:
                # Remove the order number and the parenthesis
                line = line.split(')', 1)[1].strip()
            else:
                # Remove the order number and the period
                line = line.split('.', 1)[1].strip()
            
            # Remove the hashtag and everything after it
            line = line.split('#', 1)[0].strip()
            filtered_lines.append(line)

    with open(output_file, 'w') as file:
        file.write('\n'.join(filtered_lines))


def clear_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'w') as file:
                file.write('')
            print(f"Cleared file: {file_path}")
        else:
            print(f"File not found: {file_path}")

def execute_commands(commands):
    """Execute a list of commands."""
    for command in commands:
        try:
            # Check if the command is a directory navigation command
            if command.startswith("cd "):
                directory = command[3:].strip()
                os.chdir(directory)
                print(f"Changed directory to: {os.getcwd()}")
            else:
                # Execute the command using subprocess
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                print(f"Command executed successfully: {command}")
                print("Output:")
                print(result.stdout)
                print("Error (if any):")
                print(result.stderr)
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"Command execution failed: {command}")
            print("Error:")
            print(str(e))

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=50,
        n=1,
        temperature=0.01,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})
   
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []
for content in vault_content:
    response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(response["embedding"])

# Convert to tensor and print embeddings
print("Converting embeddings to tensor...")
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)

# Conversation loop
print("Starting conversation loop...")
conversation_history = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text, and then using it to provide a list of commands that will achieve the given directive. You will determine if the action is a macro or a Shell Command. You will format your response in a way that will inform a second AI which will begin building the command. If it is a Macro, you will refer to your context. If it is a shell command, you also will refer to your context. You are to transform the query of the user into an ordered list of them commands that can complete said query. You do not bring in extra relevant infromation to the user query from outside the given context, nor the relvelant documents at hand. You are to determine the commands to make, and list them. Example: 1.) COMMAND1 #<description> 2.) COMMAND2 #<description> 3.) COMMAND3 #<description>"


#THESE ARE NOTES FOR VERSION 4
#MICROPHONE USAGE TO SPEAK TO AI

#NEW FUNCTION V4 TWO BELOW
def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def wait_for_button_press():
    input("Press Enter to run the script...")

def record_and_transcribe():
    # Record audio for 10 seconds
    record_command = ['arecord', '-D', 'hw:2,0', '-c', '1', '-f', 'S16_LE', '-r', '16000', '-t', 'wav', '-d', '10', '/home/jimbo/Desktop/RAGV5/output.wav']
    subprocess.run(record_command, check=True)

    # Transcribe the recorded audio
    transcribe_command = ['whisper', '/home/jimbo/Desktop/RAGV5/output.wav', '--language', 'en', '--task', 'transcribe', '--model', 'tiny']
    subprocess.run(transcribe_command, check=True)

while True:
    wait_for_button_press()
    record_and_transcribe()
    
    # Read the text from output.txt
    user_input = read_text_from_file('/home/jimbo/Desktop/RAGV5/output.txt')
    
    if user_input.lower() == 'quit':
        break
    
    # First call to ollama_chat with the initial system message
    response1 = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "Response 1: \n\n" + response1 + RESET_COLOR)
    
    # Update the system message for the second call
    system_message2 = """Based on the previous response, double check that the command passed is in one of the following formats:
		"Shell Command Example: 1.) COMMAND1 #<description> 2.) COMMAND2 #<description> 3.) COMMAND3 #<description>"
		"Custom Application format: "python3 actionV3.py --app lowercasename --action name --content "" """ 
    
    # Pass the response from the first call as input to the second call, along with the updated system message
    response2 = ollama_chat(response1, system_message2, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "Response 2: \n\n" + response2 + RESET_COLOR)
    
    # Save the final response to a file
    save_response_to_file(response2, '/home/jimbo/Desktop/RAGV5/listfromRag.txt')
    
    # Process the response file and create the filtered output
    process_file('/home/jimbo/Desktop/RAGV5/listfromRag.txt', '/home/jimbo/Desktop/RAGV5/outputFILTER.txt')
    
    # Read the filtered commands from the file
    with open('/home/jimbo/Desktop/RAGV5/outputFILTER.txt') as file:
        commands = file.readlines()
    
    # Strip leading/trailing whitespace and remove empty lines
    commands = [command.strip() for command in commands if command.strip()]
    
    # Execute the commands
    execute_commands(commands)
    
    # Clear the contents of listfromRag.txt and outputFILTER.txt
    clear_files(['/home/jimbo/Desktop/RAGV5/listfromRag.txt', '/home/jimbo/Desktop/RAGV5/outputFILTER.txt']) 

