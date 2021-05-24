#sudo apt install ffmpeg
ffmpeg \
-i s2_out.mp4 \
-r 15 \
-vf scale=512:-1 \
-ss 00:00:03 -to 00:00:08 \
s2_out_demo_k400.gif
