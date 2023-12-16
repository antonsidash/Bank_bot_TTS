import asyncio

from handlers import commands
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
import config
import logging


async def main():
    print('Bot online')

    #Инициализация роутеров
    dp.include_router(commands.router)
    await dp.start_polling(bot)

storage = MemoryStorage()
logging.basicConfig(level=logging.INFO)
bot = Bot(token=config.token)
dp = Dispatcher()





if __name__ == '__main__':
    asyncio.run(main())