package Java;
import java.util.Random;
import java.util.Scanner;

class Player {
	
	public String firstName;
	public String lastName;
	
	Player(String a, String b){
		firstName = a;
		lastName = b;
	}
	
	public String getRegularOrder(){
		return firstName + " " + lastName;
	}
	
	public String getReverseOrder(){
		return lastName + ", " + firstName;
	}
}

class Team {
	public static void showPlayer() {
		Player a = new Player("John", "Smith");
		Player b = new Player("Jane", "Rogers");
		
		System.out.println("Player #1 is "+a.getRegularOrder()+".");
		System.out.println("Player #2 is "+b.getRegularOrder()+".");
		System.out.println("Player #1 is "+a.getReverseOrder()+".");
		System.out.println("Player #2 is "+b.getReverseOrder()+".");
	}
}

class Rectangle {
	
	public double length;
	public double width;
	
	Rectangle(){
		length = 5;
		width = 3;
	}
	
	Rectangle(double a, double b){
		length = a;
		width = b;
		
	}
	
	public double getArea(){
		return length*width;
	}

}

class TestRectangle {
	
	public static void showRectangle(){
		Rectangle a = new Rectangle();
		Rectangle b = new Rectangle(100,200);
		System.out.println("The area of the rectangle of length "+String.valueOf(a.length)+" and width "+String.valueOf(a.width)+" is "+String.valueOf(a.getArea()));
		System.out.println("The area of the rectangle of length "+String.valueOf(b.length)+" and width "+String.valueOf(b.width)+" is "+String.valueOf(b.getArea()));
	}

}

public class Introduction {
	public static void main(String[] args) {
		/*
		 * Team.showPlayer();
		 * TestRectangle.showRectangle();
		 */
		Random r = new Random();
		Scanner s = new Scanner(System.in);
		int n = s.nextInt(), count = 0;
		for (int i = 0; i < 100000; i ++) {
			int[] a = new int[n];
			System.out.print(i + ": ");
			for (int j = 0; j < n; j++) {
				a[j] = r.nextInt(365);
				System.out.print(a[j] + " ");
			}
			for (int j = 0; j < n; j++) {
				for (int k = 0; k < j; k++) {
					if (a[j] == a[k]) {k = j = n; count++; System.out.print("bad");}
				}
			}
			System.out.println();
		}
		System.out.print(count);
		s.close();
	}
}
