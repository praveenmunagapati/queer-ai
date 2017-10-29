import tensorflow
from api import chat
from seq2seq import decoder

def main(_):
    import os
    import sys

    from twisted.python import log
    from twisted.internet import reactor
    log.startLogging(sys.stdout)

    with tensorflow.Session() as sess:
        d = decoder.Decoder(sess)
        factory = chat.ChatBotFactory()
        factory.setDecoder(d)

        reactor.listenTCP(os.getenv('PORT', 9000), factory)
        reactor.run()

if __name__ == "__main__":
    tensorflow.app.run()
