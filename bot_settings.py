import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
import graf
import logging
import random
import re
import json

i = 0
logging.basicConfig(format = u'%(levelname)-8s [%(asctime)s] %(message)s', level = logging.DEBUG, filename = u'mylog.log')
'''logger = logging.getLogger(__name__)
logger_handler = logging.FileHandler("sample.log")
logger_handler.setLevel(logging.INFO)
logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
logger.addHandler(logger_handler)'''



def write_msg(user_id, message):
    vk.messages.send(user_id=user_id, random_id=random.randint(-10000000, 100000000), message=message)
    logging.info("Ответ: " + " user_id: " + str(user_id), " message: " + str(message))
    #vk.method('message.send', {'user_id':user_id, 'message':message})

def write_msg_keyword(user_id, message, keyboard):
    vk.messages.send(user_id=user_id, random_id=random.randint(-10000000, 100000000), message=message, keyboard=keyboard)
    logging.info("Ответ: " + " user_id: " + str(user_id), " message: " + str(message))
    #vk.method('message.send', {'user_id':user_id, 'message':message})


def open_data():
    data_file = graf.text_file()
    return data_file

token = "608ecae2446e52b11badce3da7f63883c6198cce7d8487c79b6802e9cfea2fadef577945ce849bf2ade4a"

vk_session = vk_api.VkApi(token=token)
vk = vk_session.get_api()

longpoll = VkLongPoll(vk_session)

i = 100000

def words_data(longpoll):
    words_rhase = ""
    words_otvet = ""
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW:

            if event.to_me:
                words_rhase = event.text
                write_msg(event.user_id, "Добавьте ответ к вашей фразе")
                break

    words_otvet = words_data_otvet(longpoll)

    return words_rhase, words_otvet

def words_data_otvet(longpoll):
    words_rhase_otvet = ""
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW:

            if event.to_me:
                words_rhase_otvet = event.text
                write_msg(event.user_id, "Ваша фраза добавлена")
                break

    return words_rhase_otvet


def graf_text(text):
    text_rhase_graf = graf.pred_morphe([text])
    text_graf_sp1, text_graf_sp2, text_graf_sp3 = graf.create_graf(text_rhase_graf[0])
    text_graf_phase_pred = graf.search_graf(text_graf_sp1, text_graf_sp2, text_graf_sp3)

    return text_graf_phase_pred, len(text_graf_sp2)

def data_update(text_rhase, text_otvet, i):
    f = open('test2.tsv', 'a')#открываем файл для добавления записи
    graf_phase, len_graf_phase = graf_text(text_rhase)#рассчитываем среднее расстояние между вершинами для фразы
    graf_otvet, len_graf_otvet = graf_text(text_otvet)#рассчитываем среднее расстояние между вершинами для ответов

    if graf_otvet == 0.0:
        graf_otvet = 1.0
    if graf_phase == 0.0:
        graf_phase = 1.0

    text_file = '\n' + str(i) + '\t' + text_rhase + '\t' + str(graf_phase) + '\t' + text_otvet + '\t' + str(len_graf_otvet) + '\t' + str(graf_otvet) + '\t' + 'good'
    f.write(text_file)
    f.close()

    i = i + 1
    return i

def keyboard():
    data = {"one_time": True,
            "buttons": [
                [{
                    "action": {
                        "type": "text",
                        "label": "Add words"
                    },
                    "color": "primary"
                },
                    {
                        "action": {
                            "type": "text",
                            "label": "Cancel"
                        },
                        "color": "default"
                    }
                ]
            ]}

    return data

while True:
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW:

            if event.to_me:
                request = event.text

                if request == 'Add words':
                    write_msg(event.user_id, "Добавьте фразу")
                    rhase, otvet = words_data(longpoll)
                    i = data_update(rhase, otvet, i)
                    print('i = ', i)
                else:
                    logging.info("Сообщение от user_id: " + str(event.user_id), " message: " + str(request))
                    request_re = re.sub("^\s+|\n|\r|\s+$", '', request)
                    data_file = open_data()
                    text = graf.screach_claster(request_re, data_file)
                    data_len = len(data_file) - (len(data_file)/2)
                    text_i = graf.screach_claster(request_re, data_file[int(data_len):])
                    print('text_i', text_i)
                    data = keyboard()
                    if text == 'Нет для вас ответа к сожалению' and text_i == 'Нет для вас ответа к сожалению':
                        write_msg_keyword(event.user_id, text, json.dumps(data))
                    elif text_i != 'Нет для вас ответа к сожалению' and text_i!='' and text == 'Нет для вас ответа к сожалению':
                        write_msg_keyword(event.user_id, text_i, json.dumps(data))
                    else:
                        write_msg_keyword(event.user_id, text, json.dumps(data))







