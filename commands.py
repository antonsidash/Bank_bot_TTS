from aiogram import F, Router
from aiogram.filters import Command
from aiogram import types,Bot

from main import bot
from aiogram.fsm.context import FSMContext
import emoji
router = Router()
import os


from aiogram.fsm.context import FSMContext
from aiogram.filters.state import State, StatesGroup
import numpy as np
import tensorflow as tf

import yaml
import os
from tensorflow import keras
from tensorflow.keras import preprocessing, utils


class StateNow(StatesGroup):
    start = State()
    ongoing = State()
    

def load_data(files_list,dir_path):
    questions, answers = [], []
    for filepath in files_list:
        file_ = open(dir_path + os.sep + filepath, 'rb')
        docs = yaml.safe_load(file_)
        conversations = docs['conversations']
        for con in conversations:
            if len(con) > 2:
                questions.append(con[0])
                replies = con[1:]
                ans = ''
                for rep in replies:
                    ans += ' ' + rep
                answers.append(ans)
            elif len(con) > 1:
                questions.append(con[0])
                answers.append(con[1])

    answers_with_tags = []
    for i in range(len(answers)):
        if isinstance(answers[i], str):
            answers_with_tags.append(answers[i])
        else:
            questions.pop(i)

    answers = ['<START> ' + answer + ' <END>' for answer in answers_with_tags]

    return questions, answers

def create_tokenizer(questions, answers):
    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(questions + answers)
    return tokenizer

def preprocess_data(tokenizer, questions, answers):
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')
    encoder_input_data = np.array(padded_questions)

    tokenized_answers = tokenizer.texts_to_sequences(answers)
    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
    decoder_input_data = np.array(padded_answers)

    tokenized_answers = tokenizer.texts_to_sequences(answers)
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
    onehot_answers = utils.to_categorical(padded_answers, len(tokenizer.word_index) + 1)
    decoder_output_data = np.array(onehot_answers)

    return encoder_input_data, decoder_input_data, decoder_output_data

def create_model(VOCAB_SIZE, maxlen_questions, maxlen_answers):
    encoder_inputs = tf.keras.layers.Input(shape=(maxlen_questions,))
    encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(maxlen_answers,))
    decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax)
    output = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_inference_models(model, VOCAB_SIZE):
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[5]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(model.layers[3](decoder_inputs), initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[6]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

def preprocess_input(tokenizer, input_sentence, maxlen_questions):
    tokens = input_sentence.lower().split()
    tokens_list = []
    for word in tokens:
        tokens_list.append(tokenizer.word_index[word])
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')

async def chat_with_bot(enc_model, dec_model, tokenizer, tests, maxlen_answers,maxlen_questions):
    states_values = enc_model.predict(preprocess_input(tokenizer,tests[0],maxlen_questions))
    empty_target_seq = np.zeros((1 , 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += f' {word}'
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            
        empty_target_seq = np.zeros((1 , 1))  
        empty_target_seq[0 , 0] = sampled_word_index
        states_values = [h , c] 
    decoded_translation = decoded_translation.split(' end')[0]
    return decoded_translation[1].upper() + decoded_translation[2:]





async def sleep_bot(message:types.Message,bot:Bot,id):
    
    
    f,info_message = await start_ai(message)
    if f:
        await bot.delete_message(chat_id=message.chat.id, message_id=id)
        await message.answer(info_message)
    # await message.delete()
    
async def start_ai(message:types.Message):
    dir_path = 'Primer'
    files_list = os.listdir(dir_path + os.sep)
    questions, answers = load_data(files_list,dir_path)

    # Create tokenizer
    tokenizer = create_tokenizer(questions, answers)

    # Preprocess data
    encoder_input_data, decoder_input_data, decoder_output_data = preprocess_data(tokenizer, questions, answers)

    # Create model
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    maxlen_questions = encoder_input_data.shape[1]
    maxlen_answers = decoder_input_data.shape[1]

    model = keras.models.load_model('model_checkpoint.h5') 

    # Create inference models
    encoder_model, decoder_model = create_inference_models(model, VOCAB_SIZE)

    # Chat with the bot
    tests = [message.text]
    return True, await chat_with_bot(encoder_model, decoder_model, tokenizer, tests, maxlen_answers,maxlen_questions)



@router.message(Command(commands=['start']))
async def start(message:types.Message,state: FSMContext):

    await message.answer('Здравствуйте'+emoji.emojize(":raised_hand:")+', задайте свой вопрос нейросети, по поводу услуг банков'+emoji.emojize(":bank:"))

@router.message(F.text)
async def chating(message:types.Message,state: FSMContext):

    await message.answer("Бот генерирует ответ ...")
    await sleep_bot(message,bot,message.message_id+1)

    
    






# @router.message(F.location)
# async def location(message:types.Message):
#     await message.delete()
#     lat = message.location.latitude
#     lon = message.location.longitude
#     state = 0
#     user_info_update(message.from_user.id, lat, lon)
#     weather = await get_weather(message.from_user.id,lat,lon)
#     user_state.update({message.from_user.id: state})

#     text = text_render(state, weather)

#     await message.answer(text, reply_markup=inline_keyboard_builder(state))