package Java;
import java.io.*;
import java.util.*;

public class GPSInfoProcessor {

	public final ArrayList<DataInstance> list = new ArrayList<DataInstance>();
  
	public GPSInfoProcessor(String fileName) {
		try{
			BufferedReader br = new BufferedReader(new FileReader(fileName));
			String s = br.readLine();
			final String[] headers = s.replaceAll(" ", "").split(",");
			while (br.ready()) {
				s = br.readLine();
				String[] l = s.replaceAll(" ", "").split(",");
				double[] args = new double[19];
				for(int i = 0; i < 19; i++) {
					try {
						args[i] = Double.valueOf(l[i]);
					}
					catch (Exception e) {args[i] = -1;}
				}
				list.add(new DataInstance(headers, args));
			}
			
			// Sample output1
			for (int i = 0; i < list.size(); i++)
				list.get(i).print();
			
			// Sample output2
			System.out.println(list.get(3).findArgs("field.longitude"));
			
			br.close();
		}
		catch(IOException e) {System.out.println("file does not exist");}
	}
	
	public static void main(String[] args) {
		new GPSInfoProcessor("gps_fix_psr.txt");
	}
  
	public static class DataInstance {
    
		public String[] headers = new String[19];
		public double[] args = new double[19];
    
		public DataInstance(String[] headers, double[] args) {
			this.headers = headers;
			this.args = args;
		}
    
		public double findArgs(String s){
			for (int i = 0; i < 19; i++) {
				if(headers[i].equals(s)) return args[i];
			}
			return -1;
		}
		
		public void print() {
			System.out.println("Printing Data Instance");
			for (int i = 0; i < 19; i++) {
				System.out.println(headers[i] + "\t" + args[i]);
			}
		}
    
	}
  
}