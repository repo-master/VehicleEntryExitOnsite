
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCDataChannel
from aiortc.exceptions import InvalidStateError
import logging
import asyncio
from pyee.asyncio import AsyncIOEventEmitter

class DataChannelHandler(AsyncIOEventEmitter):
    def __init__(self):
        super().__init__()
        self._channels = list()
    def setupChannel(self, channel : RTCDataChannel, peer : RTCPeerConnection):
        channel.on('message', lambda msg: self.emit("data", msg, channel, peer))
        if channel.readyState == "open":
            self.emitChannelConnected(channel)
        else:
            channel.on('open', lambda: self.emitChannelConnected(channel))
        self._channels.append(channel)
    def broadcast(self, msg):
        for chn in self._channels:
            try:
                chn.send(msg)
            except InvalidStateError:
                pass
    def emitChannelConnected(self, channel : RTCDataChannel):
        self.emit("connect", channel)

class RTCPeer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._incoming_datachannel_handlers = dict()
        self._mediatracks = list()

    def addInDataChannelHandler(self, **handler : DataChannelHandler):
        """
        Incoming channel handler - the data channels created by a peer to us
        will be handled using the handlers passed to this function
        """
        for label, handler in handler.items():
            if label not in self._incoming_datachannel_handlers:
                self._incoming_datachannel_handlers[label] = []
            self._incoming_datachannel_handlers[label].append(handler)
        return self
    
    def addMediaTrack(self, track_name : str, track_instance : MediaStreamTrack):
        if not isinstance(track_instance, MediaStreamTrack):
            raise TypeError("The parameter 'track_instance' must be an object of type \"MediaStreamTrack\", but got \"%s\"" % type(track_instance))
        self._mediatracks.append(track_instance)
        return self
    
class RTCServer(RTCPeer):
    """
    An RTC connection server that receives interested peers and saves them
    in a list, and serves them content such as video streams, audio streams
    and data channel I/O.
    Works along with aiohttp to accept offer requests 
    """

    def __init__(self, app : web.Application, offer_endpoint : str = '/offer'):
        super().__init__()
        self.__app = app

        #List of peer connections that are made
        self.__pc_list = set()

        #Set-up routes for the WebRTC signalling
        self.__app.router.add_post(offer_endpoint, self.__peer_offer_negotiation)

        #Procedure to perform clean-up
        self.__app.on_shutdown.append(self.__on_app_shutdown_handler)
        

    async def _populatePCOffer(self, pc : RTCPeerConnection):
        """
        Add data channels and AV tracks to the peer connection
        """

        #Client-generated data channel handlers
        @pc.on('datachannel')
        async def on_client_dc_create(channel):
            try:
                for hdl in self._incoming_datachannel_handlers[channel.label]:
                    hdl.setupChannel(channel, pc)
            except KeyError:
                self.logger.warn("No handlers were registered for incoming channel \"%s\"" % channel.label)
                
        #Add media tracks
        for track in self._mediatracks:
            pc.addTrack(track)

    async def __peer_offer_negotiation(self, request : web.Request) -> web.Response:
        """
        An RTC client will request for supported data on the RTC,
        so we need to give a json representation of that.
        Also, this is where we begin our peer connection
        """

        self.logger.debug("Got a client RTC offer from <%s>" % request.remote)

        #Convert given parameters to an RTCSessionDescription object to manipulate later
        params = await request.json()
        if 'sdp' not in params or 'type' not in params:
            return web.json_response({"message": "Invalid data sent"}, status=400)
        
        #Construct the SessionDescription (offer) object
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

        #Create an RTCPeerConnection object for this client. This connection will remain
        #till the client disconects or times out
        pc = RTCPeerConnection()
        self.__pc_list.add(pc)
        pc._consumers = []
        await pc.setRemoteDescription(offer)
        
        #Monitor client's connection state. Remove from the list if disconnected
        @pc.on('connectionstatechange')
        async def on_pc_connstatechange():
            self.logger.debug("Connection state of <%s> changed to: %s" % (request.remote, pc.connectionState))
            if pc.connectionState == "failed":
                await pc.close()
            if pc.connectionState == "closed":
                self.logger.info("Connection to <%s> closed" % request.remote)
                self.__pc_list.discard(pc)

        #Set-up our data and AV channels
        await self._populatePCOffer(pc)

        #Create an answer based on what media we can provide...
        await pc.setLocalDescription(await pc.createAnswer())

        #...and send it to the client
        return web.json_response({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        })

    async def __on_app_shutdown_handler(self, app):
        self.logger.info("Waiting for peer connections to close...")
        await asyncio.gather(*[pc.close() for pc in self.__pc_list])
        self.__pc_list.clear()
