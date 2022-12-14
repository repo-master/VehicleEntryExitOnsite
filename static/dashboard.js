
const timeFromUpdateList = [];

//The things you need to do when you don't use React...
class LogMessage extends HTMLLIElement  {
    constructor(data) {
        super();
        this.classList = "d-flex justify-content-between mb-4";

        const msgTime = dayjs(data.ts);

        //Add card with body and time
        const msgCard = document.createElement('div'); msgCard.classList="card w-100";
        const msgCardHeader = document.createElement('div'); msgCard.appendChild(msgCardHeader);
        msgCardHeader.classList="card-header d-flex justify-content-between p-3 bg-info";
        const msgHeaderTitle = document.createElement('p'); msgCardHeader.appendChild(msgHeaderTitle);
        msgHeaderTitle.classList = "fw-bold mb-0";
        msgHeaderTitle.innerHTML = data.title;
        const msgHeaderTS = document.createElement('p'); msgCardHeader.appendChild(msgHeaderTS);
        msgHeaderTS.classList = "text-dark small mb-0 text-with-tooltip";
        msgHeaderTS.title = msgTime.format();
        msgHeaderTS.innerHTML = msgTime.fromNow();
        timeFromUpdateList.push({"e": msgHeaderTS, "djs": msgTime})

        const msgCardBody = document.createElement('div'); msgCard.appendChild(msgCardBody);
        msgCardBody.classList="card-body";
        const msgCardBodyText = document.createElement('p'); msgCardBody.appendChild(msgCardBodyText);
        msgCardBodyText.classList = "mb-0";
        msgCardBodyText.innerHTML = data.message;

        this.appendChild(msgCard);
    }
}
customElements.define("log-item", LogMessage, { extends: "li" });

setInterval(() => {
    timeFromUpdateList.forEach(i => {
        i.e.innerHTML = i.djs.fromNow();
    });
}, 30000);

//Sends the local description to other peer and wait for reply offer
const offerSignalling = pc => fetch('/offer', {
        body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type
        }),
        headers: {
            'Content-Type': 'application/json'
        },
        method: 'POST'
    })
    .then(data => data.json())
    .then(answer => pc.setRemoteDescription(answer));

const negotiateOffer = (pc) => pc.createOffer({
      offerToReceiveAudio: 0,
      offerToReceiveVideo: 1
    })
    .then(localOffer => pc.setLocalDescription(localOffer))
    .then(() => new Promise((resolve, reject) => {
        /* Setting the local offer triggers ICE gathering.
           Wait for ICE gathering to complete (search possible network paths) */
        function checkIfComplete() {
            if (pc.iceGatheringState === 'complete') {
                pc.removeEventListener('icegatheringstatechange', checkIfComplete);
                resolve();
            }
        }
        pc.addEventListener('icegatheringstatechange', checkIfComplete);
        
        //If it completed early, immediately resolve
        if (pc.iceGatheringState === 'complete')
            resolve();
    }))
    .then(() => offerSignalling(pc));

const pcClient = new RTCPeerConnection();

function start(pc) {
    //Main video
    pc.addTransceiver("video", {
        direction: "recvonly"
    }).dstVid = document.getElementById('video_main');
    //Sub-video
    pc.addTransceiver("video", {
        direction: "recvonly"
    }).dstVid = document.getElementById('video_sub');
    //Events
    const log_dc = pc.createDataChannel('log');
    const status_dc = pc.createDataChannel('status');
    
    function addLogMessage(elem) {
        const logHolder = document.getElementById('vehicle_events');
        logHolder.insertBefore(elem, logHolder.children[0]);
    }

    log_dc.onmessage = function(evt) {
        const data = JSON.parse(evt.data);
        const el = document.getElementById('vehicle_history');
        let content = "";
        for (let row of data) {
            //console.log(row)
            //const entry_ts = dayjs(row.entry_state_ts);
            //const exit_ts = dayjs(row.exit_state_ts);
            const duration = row.presence_duration || '-';

            content += "<tr>";
            content += "<td>";
            content += row.plate_number;
            content += "</td>";
            content += "<td>";
            content += row.type;
            content += "</td>";
            content += "<td>";
            content += row.entry_state_ts;
            content += "</td>";
            content += "<td>";
            content += row.exit_state_ts;
            content += "</td>";
            content += "<td>";
            content += duration;
            content += "</td>";
            content += "</tr>";
        }
        el.innerHTML = content;
    };

    status_dc.onmessage = function(evt) {
        const data = JSON.parse(evt.data);
        const el = document.getElementById('vehicle_detection');
        console.log(data)
        let content = "";
        for (let row of data) {
            const appear_ts = dayjs(row.last_state_timestamp);
            content += "<tr>";
            content += "<td>";
            content += row.plate_number;
            content += "</td>";
            content += "<td>";
            content += row.type;
            content += "</td>";
            content += "<td>";
            content += row.entry_exit_state;
            content += "</td>";
            content += "<td>";
            content += appear_ts.format();
            content += "</td>";
            content += "</tr>";
        }
        el.innerHTML = content;
    };

    pc.addEventListener('track', function(evt) {
        if (evt.track.kind == 'video') {
            const txc = evt.transceiver;
            if (txc.dstVid)
                txc.dstVid.srcObject = evt.streams[0];
        }
    });
        
    negotiateOffer(pcClient);
}

window.addEventListener('DOMContentLoaded', e => start(pcClient))
