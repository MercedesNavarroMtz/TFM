import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async


class NodesConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        ''' Cliente se conecta '''

        # Se une a la sala
        #print(self.scope['url_route']['kwargs']['room_name'])
        #self.room_group_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'nodes_control'

        # Se une a la sala
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        # Informa al cliente del éxito
        await self.accept()

    async def disconnect(self, close_code):
        ''' Cliente se desconecta '''
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data):
        ''' Cliente envía un ack y nosotros lo recibimos '''

        text_data_json = json.loads(text_data)
        # await sync_to_async(saveNodeLogMessage)(text_data_json)

        date_time = text_data_json["date_time"]
        node_name = text_data_json["node_name"]
        msg_type = text_data_json["msg_type"]
        msg = text_data_json["msg"]

        await self.channel_layer.group_send(
            'nodes_control',
            {
                "type": "send.message",
                "date_time": date_time,
                "node_name": node_name,
                "msg_type": msg_type,
                "msg": msg,
            },
        )

    async def send_message(self, event):
        ''' Recibimos información de la sala '''
        date_time = event["date_time"]
        node_name = event["node_name"]
        msg_type = event["msg_type"]
        msg = event["msg"]
    
        #Send message to WebSocket
        #await self.send(text_data=event)
        await self.send(
            text_data=json.dumps(
                {
                    "type": "send.message",
                    "date_time": date_time,
                    "node_name": node_name,
                    "msg_type": msg_type,
                    "msg": msg,
                }
            )
        )