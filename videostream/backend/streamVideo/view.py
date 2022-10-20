from copyreg import constructor
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
# from streamVideo.camera import VideoCamera
from coreAI.detect import Detect
detect = Detect()
# Create your views here.
def index(request):
    return render(request, 'index.html')
def gen():
    while True:
        frame = detect.detect()
        for i in frame:
            print('Fire > ', i[1])
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + i[0] + b'\r\n\r\n')
def video_stream(request):
    return StreamingHttpResponse(gen(),
                    content_type='multipart/x-mixed-replace; boundary=frame')