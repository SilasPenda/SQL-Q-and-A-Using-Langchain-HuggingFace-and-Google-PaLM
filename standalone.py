from utils.langchain_utils import LangchainHelper
import argparse


if __name__ == "__main__":
    # Create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--few_shot", action="store_true", help="Enable few shot learning")
    args = parser.parse_args()

    use_few_shot = args.few_shot

    langchain_helper = LangchainHelper()

    if use_few_shot:
        print("Few shot learning enabled!")
        chain = langchain_helper.get_db_chain(use_few_shot=True)
    else:
        chain = langchain_helper.get_db_chain()

    question = input("Ask a question: ")
    answer = chain.run(question)

    print("Answer: ", answer)

    