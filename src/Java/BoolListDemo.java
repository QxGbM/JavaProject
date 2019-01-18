package Java;
interface BoolList {
	public int size();
	public void insert(int i, boolean value);
	public void remove(int i);
	public boolean lookup(int i);
	public void negateAll();
	public String toString();
}

class BoolArrayList implements BoolList{
	
	private boolean[] array;
	
	public BoolArrayList() {
		array = new boolean[0];
	}
	
	public int size() {
		return array.length;
	}

	public void insert(int i, boolean value) {
		boolean[] t = new boolean[size() + 1];
		int j = 0;
		while (j < i && j < size()){
			t[j] = array[j]; 
			j = j+1;
		}
		t[j] = value;
		while (j < size()) {
			t[j+1] = array[j]; j = j+1;
		}
		array = t;
		return;
	}

	public void remove(int i) {
		if (size() == 0 || i >= size()) return;
		boolean[] t = new boolean[size() - 1];
		int j = 0;
		while(j < i && j < size()) {
			t[j] = array[j];
			j = j+1;
		}
		while (j < size()-1) {
			t[j] = array[j+1]; j = j+1;
		}
		array = t;
		return;
	}

	public boolean lookup(int i) {
		if (i < size()) return array[i];
		else return false;
	}

	public void negateAll() {
		for (int i = 0; i < size(); i = i+1) 
			if (array[i]) array[i] = false; else array[i] = true;
	}
	
	public String toString() {
		String result = "";
		for (int i = 0; i < size(); i = i+1)
			if (array[i]) result += "1"; else result += "0";
		return result;
	}
}

class BoolLinkedListNode{
	
	public boolean value;
	
	public BoolLinkedListNode next;
	public BoolLinkedListNode prev;
	
	public BoolLinkedListNode(boolean v){
		value = v;
		next = null;
		prev = null;
	}
}

class BoolLinkedList implements BoolList{
	
	private int size;
	private BoolLinkedListNode head;

	public BoolLinkedList() {
		head = null;
		size = 0;
	}
	
	public int size() {
		return size;
	}
	
	public void insert(int i, boolean value) {

		BoolLinkedListNode t = head;
		BoolLinkedListNode n = new BoolLinkedListNode(value);
		if(head == null) {
			head = n;
			size = size + 1;
			return;
		}
		else if (i == 0) {
			head = n;
			t.prev = n;
			n.next = t;
			size = size + 1;
			return;
		}
		else while(i > 0) {
			if(t.next == null) {
				t.next = n;
				n.prev = t;
				size = size + 1;
				return;
			}
			t = t.next;
			i = i - 1;
		}
		t.prev.next = n;
		n.prev = t.prev;
		t.prev = n;
		n.next = t;
		size = size + 1;
		return;
	}
	
	public void remove(int i){
		BoolLinkedListNode t = head;
		if(head == null)
			return;
		else if (i == 0) {
			head = t.next;
			t.next.prev = null;
			size = size - 1;
			return;
		}
		else while(i > 0) {
			if(t.next == null) 
				return;
			t = t.next;
			i = i - 1;
		}
		t.prev.next = t.next;
		if(t.next != null)
			t.next.prev = t.prev;
		size = size - 1;
		return;
	}
	
	public boolean lookup(int i) {
		BoolLinkedListNode t = head;
		while(i > 0)
		{
			if(t.next == null)
				return false;
			t = t.next;
			i = i - 1;
		}
		return t.value;
	}
	
	public void negateAll(){
		if(size == 0)
			{System.out.println("Empty List"); return;}
		BoolLinkedListNode t = head;
		while(t != null){
			if(t.value)	t.value = false;
			else t.value = true;
			t = t.next;
		}
		return;
	}
	
	public String toString() {
		String result = "";
		BoolLinkedListNode t = head;
		for (; t.next != null ; t = t.next)
			if (t.value) result += "1"; else result += "0";
		return result;
	}
}

public class BoolListDemo {
	
	public static int boolToSigned(BoolList lst) {
		
		int i = 1, size = lst.size(), result = 0;
		while(i<size){
			if(lst.lookup(i)) result = result * 2 + 1;
			else result = result * 2;
			i = i + 1;
		}
		if(lst.lookup(0)) return result - (int) Math.pow(2, size-1);
		else return result;
	}

	public static void main(String[] args) {

		BoolList lst = new BoolArrayList();
		// insert empty: 1
		lst.insert(0, true);
		// insert to the end: 10
		lst.insert(10000, false);
		// insert to the front: 010
		lst.insert(0, false);
		// insert to the middle: 0110
		lst.insert(1, true);
		// remove the front: 110
		lst.remove(0);
		lst.insert(0, true); // 1110 = -2
		
		System.out.println(boolToSigned(lst));
		
		lst.negateAll(); // 0001 = 1
		
		System.out.println(boolToSigned(lst));
		
		lst.remove(2); // remove in the middle
		lst.insert(2, true); // 0011 = 3 
		
		System.out.println(boolToSigned(lst));
		
		lst.remove(10000); // remove the very end 0011 =3
		
		System.out.println(boolToSigned(lst));
	}

}
