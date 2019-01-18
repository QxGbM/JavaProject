package Java;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
class HuffmanNode {
	char symbol;
	int frequency;
	HuffmanNode left;
	HuffmanNode right;
	public HuffmanNode(char c, int f)
		{ symbol = c; frequency = f; left = null; right = null; }
	public HuffmanNode(int f, HuffmanNode l, HuffmanNode r)
		{ symbol = (char)0; frequency = f; left = l; right = r; }
}
class HuffmanCode {
	char symbol;
	int frequency;
	String code;
	public HuffmanCode(char c, int f, String s)
		{ symbol = c; frequency = f; code = s; }
}
public class HuffmanCompressor {
	public static ArrayList<HuffmanNode> scanInput(String url) throws Exception{
		int a[] = new int [200], counter = 0;
		BufferedReader reader = new BufferedReader(new FileReader(new File(url)));
    	while (reader.ready()) {
    		int i = reader.read();
    		if (a[i] == 0) counter++;
    		a[i]++;
    	}
		if (reader != null) reader.close();
		ArrayList<HuffmanNode> list = new ArrayList<HuffmanNode>();
		for(int i =0 ; i < counter; i = i+1){
			int min = Integer.MAX_VALUE, t = 0;
			for(int j = 0; j < a.length; j = j+1)
				if(a[j] > 0 && a[j] < min) {min = a[j];t = j;}
			list.add(new HuffmanNode((char)t, a[t]));
			a[t]=0;
		}
		return list;
	}
	public static HuffmanNode produceTree(ArrayList<HuffmanNode> list){
		while(list.size() >= 2){
			int i;
			for (i = 2; i < list.size() && list.get(i).frequency < list.get(0).frequency + list.get(1).frequency; i = i+1) {}
			list.add(i, new HuffmanNode(list.get(0).frequency + list.get(1).frequency, list.get(0), list.get(1)));
			list.remove(1);
			list.remove(0);
		}
		return list.get(0);
	}
	public static ArrayList<HuffmanCode> traverseTree(HuffmanNode root, String encoding){
		ArrayList<HuffmanCode> list = new ArrayList<HuffmanCode>();
		if (root.symbol != (char) 0)
			list.add(new HuffmanCode(root.symbol, root.frequency, encoding)); 
		else {
			list.addAll(traverseTree(root.left, encoding + "0"));
			list.addAll(traverseTree(root.right, encoding + "1"));
		}
		return list;
	}
	public static int produceOutput(String url1, String url2, ArrayList<HuffmanCode> list) throws Exception{
		BufferedReader reader = new BufferedReader(new FileReader(new File(url1)));
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(url2)));
		int savings = 0;
    	while (reader.ready()) {
    		char c = (char) reader.read();
    		int i;
    		for(i = 0; i < list.size() && list.get(i).symbol != c; i = i+1) {}
    		writer.write(list.get(i).code);
    		savings += 8 - list.get(i).code.length();
    	}
		if (reader != null)	reader.close();
		if (writer != null) {writer.flush(); writer.close();}
		return savings;
	}
	public static String huffmanCoder(String inputFileName, String outputFileName){
		ArrayList<HuffmanNode> l1 = null;
		ArrayList<HuffmanCode> l2 = null;
		HuffmanNode root = null;
		int savings = 0;
		try  { l1 = scanInput(inputFileName); } 
		catch (Exception e) { return "Input file error"; } 
		finally { root = produceTree(l1); l2 = traverseTree(root, ""); }
		try { savings = produceOutput(inputFileName, outputFileName, l2); } 
		catch (Exception e) { return "File I/O error"; }
		for(int i = 0; i < l2.size(); i = i+1)
			System.out.println(l2.get(i).symbol + ": " + l2.get(i).frequency + ": " + l2.get(i).code);
		System.out.println("Space Saved (in bits): " + savings);
		return "OK";
	}
	public static void main(String[] args) {
		System.out.println(huffmanCoder(args[0], args[1]));
	}
}
