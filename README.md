# Mistral 7B fine-tuning examples

## QLoRA finetune on [samsum](https://huggingface.co/datasets/samsum) dataset
Fine-tuning completion Mistral 7B model on summarization [samsum](https://huggingface.co/datasets/samsum) dataset. Fine-tuning code is shown in `mistral-finetune-samsum.ipynb`

Use $r=8$, $\alpha=16$, with training prompt (`data_point` refers to each train item):
```
### Dialogue:
{data_point["dialogue"]}

### Summary:
{data_point["summary"]}
```

### Before fine-tuning

```
### Dialogue:
Jennifer: The new Marvel movies haven't been doing well, and I am pretty sure the new one also sucks.
Lee: But I really wanted to see how Miss Marvel would look in theater!
Zain: Well, my friend have seen Miss Marvel movie already, and they said that it really sucked. That's even considering them having all the comic books too.
Lee: Huh... Well, I guess I'll see it by myself when it hits the VOD. So, which one should we watch now?
Jennifer: There's tickets to Avatar 2 in about an hour. We could get those and watch that instead.
Lee: Sounds good to me. I really liked the first one anyways.
Zain: I'm on board also. Let's see that one instead! I'll buy the popcorn.

### Summary:
The new Marvel movies haven't been doing well, and I am pretty sure the new one also sucks. But I really wanted to see how Miss Marvel would look in theater! But my friend have seen Miss Marvel movie already, and they said that it really sucked. That's even considering them having all the comic books too. Well, I guess I'll see it by myself when it hits the VOD. So, which one should we watch now? There's tickets to Avatar 2 in about an hour. We could get those and watch that instead. I really liked the first one anyways. I'm on board also. Let's see that one instead! I'll buy the popcorn.

### Vocabulary:
- **sucks**: (v) to be bad or not good
- **sucked**: (v) to be bad or not good
- **sucking**: (v) to be bad or not good
- **sucked**: (v) to be bad or not good
- **sucked**: (v) to be bad or not good
- **sucked**: (v) to be bad or not good
- **sucked**: (v
```

### After fine-tuning

```
### Dialogue:
Jennifer: The new Marvel movies haven't been doing well, and I am pretty sure the new one also sucks.
Lee: But I really wanted to see how Miss Marvel would look in theater!
Zain: Well, my friend have seen Miss Marvel movie already, and they said that it really sucked. That's even considering them having all the comic books too.
Lee: Huh... Well, I guess I'll see it by myself when it hits the VOD. So, which one should we watch now?
Jennifer: There's tickets to Avatar 2 in about an hour. We could get those and watch that instead.
Lee: Sounds good to me. I really liked the first one anyways.
Zain: I'm on board also. Let's see that one instead! I'll buy the popcorn.

### Summary:
  Lee, Jennifer and Zain will watch Avatar 2 instead of Miss Marvel. Zain will buy the popcorn.

### Analysis:
Lee, Jennifer and Zain will watch Avatar 2 instead of Miss Marvel. Zain will buy the popcorn.

### Special notes:


### Comparison:


### Summary:
Lee, Jennifer and Zain will watch Avatar 2 instead of Miss Marvel. Z
```

Fine-tuned model succesfully adopts to the task where dialogue is condensed into a single sentence.