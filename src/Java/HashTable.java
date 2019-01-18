package Java;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
class Bucket {
	public HashNode head;
	public int n;
	public Bucket () { head = null; n = 0; }
}
class HashNode {
	public String word;
	public int count;
	public HashNode next;
	public HashNode (String s) { word = s; count = 1; next = null; }
}
public class HashTable {
	public static Bucket[] resize (Bucket[] list, int loadfactor) {
		if (loadfactor < 0) return list;
		else {
			Bucket[] oldlist = list;
			list = new Bucket[list.length * 2];
			for (int i = 0; i < list.length; i = i+1) list[i] = new Bucket();
			for (int i = 0; i < oldlist.length; i = i+1) {
				HashNode t = oldlist[i].head;
				while (t != null) {
					int h = t.word.hashCode() % list.length;
					if (h < 0) h += list.length;
					if (list[h].head == null)  
						{ list[h].head = t; list[h].n++; t = t.next; list[h].head.next = null; }
					else {
						HashNode q;
						for (q = list[h].head; q.next != null; q = q.next) {}
						q.next = t; list[h].n++; t = t.next; q.next.next = null;
					}
				}
			}
			return list;
		}
	}
	public static String wordCount(String input_file, String output_file) {
		Bucket[] list = new Bucket[16];
		for (int i = 0; i < list.length; i = i+1) list[i] = new Bucket();
		int n = 0;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(new File(input_file)));
	    	while (reader.ready()) {
	    		String[] in = reader.readLine().split(" |\\.|,|;|:|\\?|!|\"|'");
	    		for (int i = 0; i < in.length; i = i+1) {
	    			if (!in[i].equals("")) {
	    				int h = in[i].toLowerCase().hashCode() % list.length;
	    				if (h < 0) h += list.length;
	    				if (list[h].head == null) {
	    					list[h].head = new HashNode(in[i].toLowerCase()); 
	    					list[h].n++; n = n + 1;
	    					list = resize (list, n * 4 - list.length * 3);
	    				}
	    				else {
	    					HashNode t = list[h].head;
	    					while (true) {
	    						if (t.word.equals(in[i].toLowerCase())) { t.count++; break; }
	    						if (t.next == null) {
	    							t.next = new HashNode(in[i].toLowerCase()); 
	    							list[h].n++; n = n + 1;
	    							list = resize (list, n * 4 - list.length * 3);
	    							break;
	    						}
	    						t = t.next;
	    					}
	    				}
	    			}
	    		}
	    	}
			reader.close();
		} catch (Exception e) { return "Input File Error."; }
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File(output_file)));
			for (int i = 0 ; i < list.length ; i ++) if (list[i].head != null) {
				HashNode t = list[i].head;
				while (t != null) {
					writer.write("(" + t.word + " " + t.count + ") ");
					t = t.next;
				}
			}
			writer.flush(); writer.close();
		} catch (Exception e) { return "Output File Error."; }
    	double l = 0;
    	for (int i = 0; i < list.length; i = i+1) if (list[i].n != 0) l += list[i].n - 1;
    	l = l / list.length;
		return "OK; Total words: " + n 
				+ ", Hash table size: " + list.length 
				+ ", Average length of collision lists: " + l;
	}

	public static void main(String[] args) {
		System.out.println(wordCount(args[0], args[1]));
	}
}
