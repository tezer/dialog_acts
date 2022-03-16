from sys import argv

from predict import predict


def run_prediction(lines: list):
    """
    Run the prediction on the given lines.
    :param lines:
    :return:
    """
    preds = predict(lines)
    for preds_i, sentence in zip(preds, lines):
        print(f"{sentence} -> {preds_i}")


if __name__ == '__main__':
    filename = argv[1] if len(argv) > 1 else None
    if filename:
        with open(filename) as f:
            lines = f.readlines()
    else:
        test_sentences = ["I’ll redesign the website by next Tuesday",
                          "John will follow up with his sales team about the compensation plan",
                          "We have to setup the documentation pages ",
                          "Send me the link to this meeting after the call",
                          "Replace the broken links on the wiki",
                          "Contact the executive team and figure out if they’re ready for the expansion plan",
                          "Review the backlog before the next call",
                          "Follow-up with the Salesforce team about their documentation",
                          "I’ll finish the onboarding videos for new users"]
    run_prediction(test_sentences)

    while True:
        sentence = input("Enter a sentence: ")
        if sentence == "":
            break
        run_prediction([sentence])
