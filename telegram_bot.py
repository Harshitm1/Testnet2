import requests
from config import TELEGRAM_CONFIG
from logger import setup_logger

logger = setup_logger('telegram_bot')

class TelegramBot:
    def __init__(self):
        self.token = TELEGRAM_CONFIG['bot_token']
        self.chat_id = TELEGRAM_CONFIG['chat_id']
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        print('TELEGRAM DEBUG INIT: base_url =', self.base_url, 'chat_id =', self.chat_id)

    def send_message(self, message):
        """
        Send message to Telegram chat
        """
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            print('TELEGRAM DEBUG RESPONSE:', response.status_code, response.text)  # Debug print
            response.raise_for_status()
            logger.info(f"Telegram message sent: {message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")
            return False

    def send_trade_alert(self, trade_type, symbol, price, size):
        """
        Send formatted trade alert
        """
        emoji = "🟢" if trade_type.lower() == "buy" else "🔴"
        message = (
            f"{emoji} <b>{trade_type.upper()} Alert</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Price: {price}\n"
            f"Size: {size}"
        )
        return self.send_message(message)

    def send_error_alert(self, error_message):
        """
        Send error alert
        """
        message = f"⚠️ <b>Error Alert</b>\n\n{error_message}"
        return self.send_message(message)

def send_telegram_message(message):
    bot = TelegramBot()
    return bot.send_message(message) 