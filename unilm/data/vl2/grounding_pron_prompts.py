
pron_templates = [
    {'Question': "what is <phrase>it</phrase>{x1y1x2y2}?", 'Answer': " It is {expression}."},
    {'Question': "what is <phrase>this</phrase>{x1y1x2y2}?", 'Answer': " This is {expression}."},
    {'Question': "Can you describe <phrase>this object</phrase>{x1y1x2y2}?", 'Answer': " This object is {expression}."},  
    {'Question': "What can you say about <phrase>that item</phrase>{x1y1x2y2}?", 'Answer': " That item is {expression}."},  
    {'Question': "How would you characterize <phrase>the thing</phrase>{x1y1x2y2}?", 'Answer': " The thing is {expression}."},  
    {'Question': "Can you identify <phrase>the object</phrase>{x1y1x2y2}?", 'Answer': " The object is identified as {expression}."},  
    {'Question': "Could you describe <phrase>this</phrase>{x1y1x2y2} for me?", 'Answer': " This is {expression}."},  
    {'Question': "How would you describe <phrase>this item</phrase>{x1y1x2y2}?", 'Answer': " This item is {expression}."},  
    {'Question': "Describe <phrase>it</phrase>{x1y1x2y2}.", 'Answer': " It is {expression}."}, 
    {'Question': "Describe <phrase>this object</phrase>{x1y1x2y2}.", 'Answer': " This object is {expression}."},  
    {'Question': "Tell me about <phrase>that item</phrase>{x1y1x2y2}.", 'Answer': " That item is {expression}."},  
    {'Question': "Explain <phrase>the thing</phrase>{x1y1x2y2}.", 'Answer': " The thing is {expression}."},  
    {'Question': "Characterize <phrase>the object</phrase>{x1y1x2y2}.", 'Answer': " The object is {expression}."},  
    {'Question': "Elaborate on <phrase>it</phrase>{x1y1x2y2}.", 'Answer': " It is {expression}."},  
    {'Question': "Identify <phrase>this thing</phrase>{x1y1x2y2}.", 'Answer': " This thing is {expression}."},  
    {'Question': "Detail <phrase>the object</phrase>{x1y1x2y2}.", 'Answer': " The object is {expression}."},  
    {'Question': "Specify <phrase>the item</phrase>{x1y1x2y2}.", 'Answer': " The item is {expression}."},  
    {'Question': "Illustrate <phrase>that object</phrase>{x1y1x2y2}.", 'Answer': " That object is {expression}."},  
    {'Question': "", 'Answer': "<phrase>It</phrase>{x1y1x2y2} is {expression}"},
    {'Question': "", 'Answer': "<phrase>The object</phrase>{x1y1x2y2} can be described as {expression}."},  
    {'Question': "", 'Answer': "<phrase>This</phrase>{x1y1x2y2} appears to be {expression}."},  
    {'Question': "", 'Answer': "<phrase>That item</phrase>{x1y1x2y2} is {expression}."}, 
]

simple_pron_templates = [
    {'Question': "what is <phrase>it</phrase>{x1y1x2y2}?", 'Answer': "It is {expression}."},
    {'Question': "what is <phrase>this</phrase>{x1y1x2y2}?", 'Answer': "This is {expression}."},
    {'Question': "Describe <phrase>this object</phrase>{x1y1x2y2}.", 'Answer': "This object is {expression}."},  
    {'Question': "Identify <phrase>this object</phrase>{x1y1x2y2}.", 'Answer': "This object is {expression}."},  
    {'Question': "", 'Answer': "<phrase>It</phrase>{x1y1x2y2} is {expression}"},
    {'Question': "", 'Answer': "<phrase>The object</phrase>{x1y1x2y2} can be described as {expression}."},  
    {'Question': "", 'Answer': "<phrase>This</phrase>{x1y1x2y2} appears to be {expression}."},  
]

brief_caption_templates = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
    "Briefly explain the picture."
    "Give a short overview of the image."
    "Outline the main features of the photo."
    "Write a short summary of the picture."
    "Present a simple description of the photo."
    "Explain the picture in a few words."
]

text_templates = [
    "Extract text from the given image.",
    "Identify the text present in this image.",
    "Please recognize the text within this image.",
    "Detect and display the text from the image.",
    "Perform text recognition on the provided image.",
    "Analyze the image and return the text found.",
    "Can you read the text in this image for me?",
    "Obtain the text from the image.",
    "Capture the text present in the image.",
    "What does the text in this image say?",
    "Conduct text recognition on the image.",
    "Extract and present the text from this image.",
]

vqa_templates = [
    "{question}",
    "Q: {question} A:",
    "Question: {question} Answer:",
]