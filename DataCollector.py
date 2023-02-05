import pickle
import pandas
import numpy as np
from twitchio.ext import commands
import tensorflow as tf


def invalid_char(data, chars):
    for _ in data:
        if _ in chars[chars.index('>'):]:
            return True
    return False


def preprocess(texts, tokenizer):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return X


def next_char(tokenizer, model, text, temperature=1.0):
    x_new = preprocess([text], tokenizer)
    # y_proba = model.predict_classes(x_new)
    # return tokenizer.sequences_to_texts(y_proba + 1)[0][-1]
    y_proba = model.predict(x_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def complete_text(model, tokenizer, text, n_chars=30, temperature=1.0):
    for _ in range(n_chars):
        text += next_char(tokenizer, model, text, temperature)
        if text[-1] == '\n':
            return text
    return text


class Bot(commands.Bot):

    def __init__(self):
        self.file = 0
        self.msg = 0
        self.msg_list = []
        self.current_text = ''
        with open('chars.pickle', 'rb') as handle:
            self.chars = pickle.load(handle)
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.model = tf.keras.models.load_model('twitchRNNSave.h5')
        super().__init__(token='None', prefix='!',
                         initial_channels=['xqcow'],
                         client_id='None')

    async def event_message(self, ctx):
        if ctx.author.name.lower() == 'nightbot' or ctx.author.name.lower() == 'streamelements':
            return
        if ctx.content[0] == '!':
            return
        if len(ctx.content) >= 15:
            return
        if len(ctx.content) <= 3:
            return
        if invalid_char(ctx.content, self.chars):
            return
        self.msg += 1
        self.msg_list.append([self.msg, str(ctx.content + '\n').replace(chr(0xe0000), '')])
        print([self.msg, str(ctx.content + '\n').replace(chr(0xe0000), '')])
        if self.msg % 1000 == 0:
            df = pandas.DataFrame(self.msg_list, columns=('num', 'message'))
            df.to_csv('C:/Users/Jamie Phelps/Documents/TwicthXQC/batch' + str(self.msg) + '.csv')
            print('saved ' + 'C:/Users/Jamie Phelps/Documents/TwicthXQC/batch' + str(self.msg) + '.csv')
            self.msg_list = []


    async def event_ready(self):
        print(f'Logged in as | {self.nick}')

    # @commands.command()
    # async def hello(self, ctx: commands.Context):
    #     await ctx.send(f'Hello {ctx.author.name}!')


bot = Bot()
if __name__ == "__main__":
    bot.run()
