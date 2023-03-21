# -*- coding: utf-8 -*-
'''
Author: ZJG
Date: 2022-11-15 16:05:12
LastEditors: ZJG
LastEditTime: 2022-11-15 22:40:55
'''
import socket
import threading
import time

def main():
    while True:
        is_server = input('本电脑是否为服务端?[y/n]\r\n')
        if is_server not in ['y', 'n']:
            print('请输入y或n...')
            continue
        break
    if is_server == 'y':
        # 把本电脑设置成服务端
        # 设置ip
        ip = '0.0.0.0'
        port = 2345
        # 创建一个socket对象，指定协议
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 监听端口
        s.bind((ip, port))
        # 注意：绑定端口的号的时候尽量不要使用小于1024的
        # 监听端口，传入的参数指等待连接的最大数量
        s.listen(6)
        print('服务端开始监听...')

        def tcp_link(sock, addr):
            # 打印连接成功
            print("%s，连接成功" % str(addr))
            # 服务器发送数据到客户端
            sock.send('已成功连接上服务端...'.encode('utf-8'))
            # 循环接收客户端发来的请求数据
            while True:
                # 接收数据，每次接收1024个字节
                data = sock.recv(1024)
                if not data:
                    break
                data = data.decode("utf-8")
                if data == 'end':
                    break
                print('收到%s的信息：%s' %(addr, data))
                sock.send(("服务器已收到： %s" % data).encode("utf-8"))
            sock.close()
            print("%s，已断开连接" % str(addr))

        # 创建一个新的线程来处理TCP连接
        while True:
            sock, addr = s.accept()
            t = threading.Thread(target=tcp_link, args=(sock, addr))
            # 开启线程
            t.start()
            t.join()
    else:
        # 把本电脑设置成客户端
        # 创建一个socket对象
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 建立连接
            ip = input('请输入服务端ip:\r\n').split()[0]
            s.connect((ip, 2345))
            # 接收消息
            print(s.recv(1024).decode("utf-8"))

            for data in ['第一份数据'.encode('utf-8'), '第二份数据'.encode('utf-8'), '第三份数据'.encode('utf-8')]:
                # 发送数据
                s.send(data)
                time.sleep(1)
                print(s.recv(1024).decode('utf-8'))
            # 最后发送结束的标识
            s.send(b'end')
            # 关闭连接
            s.close()
        except Exception as e:
            print('连接失败',e)

        print(input('请输入任意键结束...'))


if __name__ == '__main__':
    main()
