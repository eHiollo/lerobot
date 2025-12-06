import socket
import json
import time
import numpy as np
import threading

class X7MockServer:
    def __init__(self, host='0.0.0.0', port=8000):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        
        # Initialize dummy state for 8 joints (7 axis + 1 gripper)
        self.q = [0.0] * 8 
        
    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5) # Allow backlog for multiple connections
        self.running = True
        print(f"X7 Server listening on {self.host}:{self.port}")
        
        try:
            while self.running:
                print("Waiting for connection...")
                client_sock, addr = self.server_socket.accept()
                print(f"Connected by {addr}")
                # Handle client in a new thread to support multiple clients (Leader + Follower)
                client_thread = threading.Thread(target=self.handle_client, args=(client_sock,))
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\nServer stopping...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def handle_client(self, conn):
        buffer = ""
        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                
                buffer += data.decode('utf-8')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    
                    self.process_command(conn, line)
                    
        except ConnectionResetError:
            print("Connection reset by peer")
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            conn.close()
            print("Client disconnected")

    def process_command(self, conn, command):
        # print(f"Received command: {command}")
        
        if command == "GET_STATE":
            # Simulate some joint movement (sine wave)
            t = time.time()
            self.q = [np.sin(t + i) * 0.1 for i in range(8)]
            
            response = json.dumps({"q": self.q}) + "\n"
            conn.sendall(response.encode('utf-8'))
            
        elif command.startswith("SET_JOINTS"):
            try:
                payload = command[len("SET_JOINTS"):].strip()
                data = json.loads(payload)
                target_q = data.get("q")
                if target_q:
                    # print(f"Setting joints to: {target_q}")
                    self.q = target_q # Update internal state to match target
            except json.JSONDecodeError:
                print("Failed to parse SET_JOINTS payload")
        
        else:
            print(f"Unknown command: {command}")

if __name__ == "__main__":
    # Default port 8000
    server = X7MockServer(port=8000)
    server.start()
