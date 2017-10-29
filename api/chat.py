import json

from autobahn.twisted.websocket import WebSocketServerProtocol, WebSocketServerFactory

class ChatBotProtocol(WebSocketServerProtocol):
    def onConnect(self, request):
        print("Client connecting: {}".format(request.peer))

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        decoder = self.factory.decoder
        if not decoder:
            self.sendMessage(json.dumps({ "message": "Loading. Please try again." }), False)
            return

        if not isBinary:
            data = json.loads(payload)
            response = decoder.decode(data["message"])
            self.sendMessage(json.dumps({ "message": response }), False)
        else:
            self.sendMessage(json.dumps({ "message": "Invalid message." }), False)

    def onClose(self, wasClean, code, reason):
        print("WebSocket was closed: {}".format(reason))

class ChatBotFactory(WebSocketServerFactory):
    protocol = ChatBotProtocol

    def setDecoder(self, decoder):
        self.decoder = decoder
