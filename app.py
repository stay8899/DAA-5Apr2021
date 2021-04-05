from aiogram import Bot, Dispatcher, types
from transformers import pipeline
from aiogram.utils import executor
import asyncio
import re
import logging
from transformers import BertTokenizer, BertForQuestionAnswering
import torch


import json
import requests
import time
import urllib
import joblib

log = logging.getLogger(__name__)

loop = asyncio.get_event_loop()
bot = Bot(token="1510968106:AAFb0GbFkVclYEalC2xLu9W0KVolK3oaW8Q")
dp = Dispatcher(bot, loop)


@dp.message_handler(commands=['start'])
async def main(message: types.Message):
    await message.reply('Welcome to QnA Bert Bot, this will work for qna from the below text, it wont work for hi , how are you etc')
    await message.reply(""" Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its share and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a privately held company on September 4, 1998. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet leading subsidiary and will continue to be the umbrella company for Alphabets Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.).""")


@dp.message_handler()
async def main(message: types.Message):

    context = """Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its share and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a privately held company on September 4, 1998. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet leading subsidiary and will continue to be the umbrella company for Alphabets Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.).."""

    modelname = "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad"
    tokenizer = BertTokenizer.from_pretrained(modelname)
    model = BertForQuestionAnswering.from_pretrained(modelname)

    mid = message.message_id
    textin = message.text

    while True:
        print('bot received a question:', textin)
        if textin == "bye" or textin == "Bye":
            reply = "Bye, see you later."
            await message.answer(reply)
            break
        else:
            #input_ids = tokenizer.encode(textin, context)
            #input_ids = inputs["input_ids"].tolist()[0]

            #encoding = tokenizer.encode_plus(text = textin, text_pair=context, add_special = True)
            # token embedding
            #inputs = encoding['input_ids']
            # 3 segment embedgin
            #sentence_embed = encoding['token_type_ids']
            # input tokens
            #tokens = tokenizer.convert_ids_to_tokens(inputs)
            #start_scores, end_scores  = model(input_ids=torch.tensor([inputs]), token_type_ids = torch.tensor([sentence_embed]))
            #start_index = torch.argmax(start_scores)
            #end_index = torch.argmax(end_scores)

            #answer = ' '.join(tokens[start_index:end_index+1])
            input_ids = tokenizer.encode(textin, context)
            #input_ids = inputs["input_ids"].tolist()[0]

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            # Search the input_ids for the first instance of the `[SEP]` token.
            sep_index = input_ids.index(tokenizer.sep_token_id)

            # The number of segment A tokens includes the [SEP] token istelf.
            num_seg_a = sep_index + 1

            # The remainder are segment B.
            num_seg_b = len(input_ids) - num_seg_a

            # Construct the list of 0s and 1s.
            segment_ids = [0]*num_seg_a + [1]*num_seg_b

            # There should be a segment_id for every input token.
            assert len(segment_ids) == len(input_ids)

            #tokens = tokenizer.convert_ids_to_tokens(input_ids)
            # Search the input_ids for the first instance of the `[SEP]` token.
            # sep_index = input_ids.index(tokenizer.sep_token_id)

            # # The number of segment A tokens includes the [SEP] token istelf.
            # num_seg_a = sep_index + 1

            # # The remainder are segment B.
            # num_seg_b = len(input_ids) - num_seg_a

            # # Construct the list of 0s and 1s.
            # segment_ids = [0]*num_seg_a + [1]*num_seg_b

            # # There should be a segment_id for every input token.
            # assert len(segment_ids) == len(input_ids)

            #answer_start_scores, answer_end_scores = model(**inputs)
            # Run our example through the model.
            start_scores, end_scores = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                                             token_type_ids=torch.tensor([segment_ids]))  # The segment IDs to differentiate question from answer_text
            # Find the tokens with the highest `start` and `end` scores.
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)

            # # Combine the tokens in the answer and print it out.
            answer = ' '.join(tokens[answer_start:answer_end+1])

            # answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            # answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

            # answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            # answer = answer.lower()
            # answer = answer.replace(" ", " ")

            #answer  = ' '.join(text_tokens[answer_start:answer_end+1])
            #reply = question_answering(question = textin, context = context)
            #replyl = re.split(r"[:]", str(answer))
            #final_reply = " ".join(re.findall(r"[a-zA-Z0-9]+", replyl[-1]))

            corrected_answer = ''

            for word in answer.split():

                # If it's a subword token
                if word[0:2] == '##':
                    corrected_answer += word[2:]
                else:
                    corrected_answer += ' ' + word
            # await message.answer(answer)
            await message.answer(corrected_answer)
        await asyncio.sleep(2)
        while True:
            if message.message_id > mid:
                mid = message.message_id
                textin = message.text
                break
            else:
                await asyncio.sleep(1)
    dp.stop_polling()
    await dp.wait_closed()
    await bot.session.close()
    log.warning('Program is ended')

if __name__ == '__main__':
    #executor.start_polling(dp, skip_updates=True)
    loop.create_task(executor.start_polling(dp, skip_updates=True))
    loop.run_until_complete(main())
    loop.stop()
    loop.close()
