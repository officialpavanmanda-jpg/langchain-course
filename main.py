from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama



def main():
    print("Hello from langchain-course!")

    introduction = """
   Thomas Michael Shelby OBE DCM MM is a fictional character and the main protagonist of the British period crime drama Peaky Blinders. He is played by Irish actor Cillian Murphy, who has won two Irish Film & Television Awards and two National Television Awards for his portrayal of Shelby. The character has received critical acclaim.

Director Steven Knight cast Cillian Murphy for the role of Tommy Shelby. The character is introduced as a First World War veteran from a family with Romani ancestry, whose criminal enterprise is centred in Birmingham. The narrative begins in 1919, and largely revolves around Shelby's romance with Grace Burgess and his conflict with Inspector Campbell. As the series progresses, Shelby's relationship with mob boss Alfie Solomons is a key element in many of the storylines.

Casting and background
Cillian Murphy had expressed an interest in doing more television roles, "Those iconic American shows had been on and we watched them and everyone was kind of conscious of that. I think the BBC were conscious of that and so I was keen to read some good scripts and they [Peaky Blinders] were the first TV scripts I got sent". Murphy admitted that he was not aware of who the Peaky Blinders were when he was initially presented with the script.[2]

Jason Statham was initially preferred for the role by director Steven Knight, who explains "I met them both in LA to talk about the role and opted for Jason. [...] Cillian, when you meet him, isn't Tommy, obviously, but I was stupid enough not to understand that". Knight then opted to cast Murphy instead after receiving a text message from Murphy that read "Remember, I'm an actor".[3]

Although many characters in the series are based on real-life historical figures, most of the Peaky Blinders are entirely fictional and were created by Knight.[4] Tommy Shelby is from a Romani family based in Birmingham.[5][6] Murphy spent time with Romani people to prepare himself for the role.[7] Shelby is a veteran of the First World War and has post-traumatic stress disorder as a result of his experiences during the war; something that is a recurring theme throughout the series.[8] Shelby sports an undercut hairstyle and this has led to a resurgence in its popularity.[9]
    """

    summary_template = """
    given the information {information} about a person, please extract the following:
    1. A short summary
    2. 2-3 key facts about the person
    """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOllama(model="gemma3:270m", temperature=0)

    chain = summary_prompt_template | llm
    response = chain.invoke({"information": introduction})
    print(response.content)

if __name__ == "__main__":
    main()
