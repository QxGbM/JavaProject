package Java;
import java.util.*;
public class PlayBattleShip{
	
	private int shipcounter;
	private int founded;
	private int[][] hitlocation;
	private int[][] shiplocation;
	
	public PlayBattleShip(int row, int col, int n){
		shipcounter = n;
		founded = 0;
		hitlocation = new int[row][col];
		shiplocation = new int[row][col];
		Random rnd = new Random();
		int i = 0;
		while (i < n) {
			int r = rnd.nextInt(row), c = rnd.nextInt(col);
			if (shiplocation[r][c] != 1) shiplocation[r][c] = 1;
			else continue;
			i = i + 1;
		}
	}
	  
	public void play(){
		Scanner input = new Scanner(System.in);
		while (founded < shipcounter){
			try {
				for(int i = 0; i < hitlocation[0].length; i++) System.out.print("\t" + i);
				System.out.println();
				for(int i = 0; i < hitlocation.length; i++) {
					System.out.print(i);
					for(int j = 0; j < hitlocation[0].length;j++)
					{
						System.out.print("\t");
						switch (hitlocation[i][j]) {
						case 1: System.out.print("X"); break;
						case -1: System.out.print("O"); break;
						case 0:  System.out.print(" "); break;
						}
					}
					System.out.println();
				}
				System.out.println("\nEnter the row number, or Q for quit, C for cheat:");
				String line = input.next();
				if (line.equals("Q")) break;
				else if (line.equals("C")) {
					System.out.println("\nLocation of the ships:\n");
					for(int i = 0; i < shiplocation.length; i++)
					for(int j = 0; j < shiplocation[0].length; j++) {
						if(shiplocation[i][j] == 1) System.out.println("row: " + i + " col: " + j);
					}
					System.out.println();
				}
				else {
			        int i = Integer.valueOf(line);
			        System.out.println("\nEnter column:");
			        int j = Integer.valueOf(input.next());
			        if (i >= hitlocation.length || j >= hitlocation[0].length)
			        	{System.out.println("\nInput out of range"); throw new Exception();}
			        else if (hitlocation[i][j] != 0)
			        	{System.out.println("\nAlready Hit"); throw new Exception();}
			        else if (shiplocation[i][j] == 1)
			        {
			            founded+=1; hitlocation[i][j] = 1;
			            System.out.println("\nHit! " + founded + " of " + shipcounter + " correct.");
			        }
			        else
			        {
			            System.out.println("\nMiss. " + founded + " of " + shipcounter + " correct.");
			            hitlocation[i][j] = -1;
			        }
				}
			} catch (Exception e) {
				System.out.println("\nPlease Enter a Valid Command\n");
				continue;
			}
		}
		if (founded == shipcounter) System.out.println("Congratulations! You Won!");
	    else System.out.print("You Gave Up!");
		input.close();
	}
	
	public static void main(String[] args){
		PlayBattleShip board = 
				new PlayBattleShip(Integer.valueOf(args[0]), Integer.valueOf(args[1]), Integer.valueOf(args[2]));
		board.play();
	}
	
  }