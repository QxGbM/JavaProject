package Networks;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class proxyd {
	
	public static class DNSBuffer {
		long time;
		String host;
		String ip;
		
		public DNSBuffer(String address) {
			// A new Buffer instance stores the time being created, the host name, and its IP.
			time = System.currentTimeMillis();
			host = address;
			try {
				ip = InetAddress.getByName(host).getHostAddress();
			} catch (UnknownHostException e) {
				e.printStackTrace();
			}
		}
	}
	
	// An Array is used to store all DNS buffer instances.
	public static ArrayList<DNSBuffer> buffer = new ArrayList<DNSBuffer>();
	
	public static String dnsLookup(String address) {
		int i = 0;
		while (i < buffer.size()) {
			long currentTime = System.currentTimeMillis();
			if (currentTime - buffer.get(i).time > 30000) {
				// If the instance in the buffer is too old, remove it.
				buffer.remove(i); continue;
			}
			if(buffer.get(i).host.equals(address)) {
				// If the host name matches the one in buffer, return the previously looked up IP.
				System.out.println("received ip from buffer");
				return buffer.get(i).ip;
			}
			i ++;
		}
		// If no record can be found, create a new instance and return the looked up IP.
		buffer.add(i, new DNSBuffer(address));
		return buffer.get(i).ip;
	}

	public static void runServer(int port) throws IOException {
		@SuppressWarnings("resource")
		// Listening input port.
		ServerSocket ss = new ServerSocket(port);
		
		while (true) {
			// Multiple Threading Implementation to guarantee progressing.
			Socket client = ss.accept();
			InputStream streamFromClient = client.getInputStream();
			OutputStream streamToClient = client.getOutputStream();
			// With each connection, start a thread.
			new Thread() {
				@Override
				public void run() {
					try {
						byte[] request = new byte[1024];
						int bytesRead = streamFromClient.read(request);
						if (bytesRead != -1) {
							String s = new String(request, 0, bytesRead);
							String[] l = s.split("\r\n");
							// First identify the request type, only GET and POST supported.
							if (l[0].substring(0,3).equals("GET") || l[0].substring(0,4).equals("POST")) {
								int i = 0;
								String host = "";
								while (i < l.length) {
									// Read in the host address.
									if (l[i].length() >= 4 && l[i].substring(0, 4).equals("Host"))
										host = l[i].replaceAll("Host: ", "");
									// Change the connection type to be closing.
									if (l[i].equals("Connection: Keep-Alive") || l[i].equals("Proxy-Connection: Keep-Alive"))
										l[i] = "Connection: Close";
									i++;
								}
								// HTTP Server uses port 80.
								int remoteport = 80;
								
								// Modify full URL into a relative URL by removing host and "http://".
								String ms = "";
								l[0] = l[0].replaceAll(host, "").replaceAll("http://", "");
								for (i = 0; i < l.length; i++)
									ms += l[i] + "\r\n";
								ms += "\r\n";
								
								// Lookup the IP address from the DNS buffer.
								String ip = dnsLookup(host);
								
								// Start the connection with the remote server.
								Socket server = new Socket(ip, remoteport);
								InputStream streamFromServer = server.getInputStream();
								OutputStream streamToServer = server.getOutputStream();
								
								// Write out the modified request.
								byte[] b = ms.getBytes();
								streamToServer.write(b, 0, b.length);
								streamToServer.flush();
								
								// Read in the response(s), and shut down the connection.
								byte[] reply = new byte[4096];
								int bytesRead2 = streamFromServer.read(reply);
								while (bytesRead2 != -1) {
									streamToClient.write(reply, 0, bytesRead2);
									streamToClient.flush();
									bytesRead2 = streamFromServer.read(reply);
								}
								server.close();
								
							}
							else {
								System.out.println("Only GET & POST Command Supported");
							}
						}
						// Close up the connection.
						client.close();
					} catch (IOException e) {e.printStackTrace();}
				}
			}.start();
		}
	}
	
	public static void main(String[] args) throws IOException {
		// Default port is 10010.
		int port = 10010;
		// -port command, change the port number.
		if (args.length > 0 && args[0].equals("-port")) port = Integer.valueOf(args[1]);
		System.out.println("Starting proxy on port " + port);
		runServer(port);
	}
}
