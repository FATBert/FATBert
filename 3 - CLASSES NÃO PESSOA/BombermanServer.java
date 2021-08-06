package Network;

import java.io.*;
import java.net.*;
import java.util.concurrent.atomic.AtomicBoolean;

import Engine.DebugConsole;


public class BombermanServer extends Thread
{
	public static void main(String args[])
	{	
		DebugConsole.setVerboseLevel(2);
		BombermanServer server = new BombermanServer(5730);
		server.setServerInputState((byte) 100);
		server.start();
		
		
		
		while (server.serverConnectedToClient() == false)
		{}
		
		
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		System.out.println(server.getClientInputState());
		server.shutdown();
	}
	
	private ServerSocket Server;
	private Socket Socket;
	private Thread Server_Thread;
	
	private int port;
	
	private boolean game_running = true;
	private AtomicBoolean connected = new AtomicBoolean(false);
	
	private volatile byte input_client;
	private volatile byte input_server;
	private Object Lock = new Object();
	
	
	public BombermanServer(int port)
	{
		DebugConsole.Print("Initializing server, port: " + port);
		
		this.port = port;
		
		DebugConsole.Print("Initializing server done");
	}
	
	
	
	public void init() throws IOException
	{
		DebugConsole.Print("Starting server");
		
		this.Server = new ServerSocket(this.port);
			
		this.Server_Thread = new Thread(this);
		this.Server_Thread.start();
		
		DebugConsole.Print("Server started");
	}
	
	public void run()
	{
		this.waitOnIncomingConnection();
			
		this.connected.set(true);
		
		while (this.game_running == true)
		{
			byte[] buffer_send = new byte[1];
			byte[] buffer_read;
			
			synchronized (this.Lock)
			{
				buffer_send[0] = this.input_server;
			}
			
			try 
			{
				this.sendPackage(buffer_send);
				buffer_read = this.getPackage();
			} 
			catch (IOException e) 
			{
				DebugConsole.PrintError("Connection to client lost");
				this.connected.set(false);
				this.game_running = false;
				break;
			}
			
			synchronized (this.Lock)
			{
				this.input_client = buffer_read[0];
			}
		}
	}
	
	public void shutdown()
	{
		DebugConsole.Print("Shutting down server...");
		
		try
		{
			this.game_running = false;
			this.Server_Thread.join();
			this.Socket.close();
			this.Server.close();
			this.connected.set(false);
		}
		catch (IOException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		DebugConsole.Print("Shutting down server successful");
	}
	
	
	
	private void sendPackage(byte[] pack) throws IOException
	{
		DataOutputStream Output = new DataOutputStream(this.Socket.getOutputStream());
		Output.write(pack);
	}
	
	private byte[] getPackage() throws IOException
	{
		byte[] buffer = new byte[1024];
		int len = 0;
		
		DataInputStream Input = new DataInputStream(this.Socket.getInputStream());
		len = Input.read(buffer);

		byte[] buffer_ret = new byte[1024];
		for (int i = 0; i < len; i++)
		{
			buffer_ret[i] = buffer[i];
		}
		
		return buffer_ret;	
	}
	
	
	
	public void setServerInputState(byte input_vector)
	{
		synchronized (this.Lock)
		{
			this.input_server = input_vector;
		}
	}
	
	public byte getClientInputState()
	{
		byte tmp;
		synchronized (this.Lock)
		{
			tmp = this.input_client;
		}
		
		return tmp;
	}
	
	public boolean serverConnectedToClient()
	{
		return this.connected.get();
	}

	private void waitOnIncomingConnection()
	{
		DebugConsole.Print("Waiting on incoming connections...");
		
		try 
		{
			this.Socket = this.Server.accept();
			DebugConsole.Print("Connection arrived");
		} 
		catch (IOException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		DebugConsole.Print("All connections arrived");
	}
	
	

	public static byte convertToPackage(boolean up, boolean down, boolean left, boolean right, boolean bomb)
	{
		byte vector = 0x00;
		if (up == true)
			vector |= (1 << 0);
		if (down == true)
			vector |= (1 << 1);
		if (left == true)
			vector |= (1 << 2);
		if (right == true)
			vector |= (1 << 3);
		if (bomb == true)
			vector |= (1 << 4);
		
		return vector;
	}
	
	public static boolean[] convertToBoolean(byte pack)
	{
		boolean[] input_states = new boolean[5];
		
		if ((pack & 1) == 1)
			input_states[0] = true;
		if ((pack & (1 << 1)) == (1 << 1))
			input_states[1] = true;
		if ((pack & (1 << 2)) == (1 << 2))
			input_states[2] = true;
		if ((pack & (1 << 3)) == (1 << 3))
			input_states[3] = true;
		if ((pack & (1 << 4)) == (1 << 4))
			input_states[4] = true;
	
		return input_states;
	}
}
