FROM library/python:3.9-slim

WORKDIR /app/VehicleEntryExitOnsite/

COPY ./requirements.txt .

#Needs git to install some packages
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install ffmpeg

#Install package dependencies
RUN pip install --upgrade pip
RUN pip install gunicorn
RUN pip install -r requirements.txt

#Copy the project
COPY . .

EXPOSE 8080

ENV HEADLESS True

CMD [ "gunicorn", "vehiclebot:VehicleEntryExitOnSite", "--bind", "0.0.0.0:8080", "--worker-class", "aiohttp.GunicornWebWorker" ]
