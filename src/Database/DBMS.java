package Database;
import java.sql.*;
import java.util.Scanner;

public class DBMS {
	
	public Connection connection;
	public Statement statement;
	public Scanner sc;
	
	public void printResult(ResultSet result) throws SQLException {
		ResultSetMetaData metadata = result.getMetaData();
		while (result.next()) {
			for (int i = 1; i <= metadata.getColumnCount(); i++) {
				System.out.print(result.getString(i));
				if (i != metadata.getColumnCount()) System.out.print("\t");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	public void Q1() throws SQLException {
		System.out.println("Enter the Singer Name (Default is Taylor Swift): ");
        String starName = sc.nextLine();
        if (starName.equals("")) starName = "Taylor Swift";
		String query = "select songTitle, songYear from Sings where starName = '" + starName + "'";
		System.out.println(query);
		ResultSet result = statement.executeQuery(query);
		System.out.println("First Query Result:");
		printResult(result);
	}
	
	public void Q2() throws SQLException {
		String query = "select Studio.studioName, songTitle, songYear from Studio, Song "
				+ "where Studio.studioName = Song.studioName "
				+ "group by Song.studioName "
				+ "having max(length)";
		System.out.println(query);
		ResultSet result = statement.executeQuery(query);
		System.out.println("Second Query Result:");
		printResult(result);
	}
	
	public void Q3() throws SQLException {
		System.out.println("Enter the Studio Name (Default is Sony): ");
        String studioName = sc.nextLine();
        if (studioName.equals("")) studioName = "Sony";
		String query = "select `name`, max(networth) from SongExec, Song "
				+ "where `producerC#` = `cert#` and studioName = '" + studioName + "'";
		System.out.println(query);
		ResultSet result = statement.executeQuery(query);
		System.out.println("Third Query Result:");
		printResult(result);
	}
	
	public void Q4() throws SQLException {
		String query = "select SongSinger.starName from SongSinger, Studio, Sings, Song "
				+ "where SongSinger.address = Studio.address "
				+ "and Sings.starName = SongSinger.starName "
				+ "and Song.songTitle = Sings.songTitle "
				+ "and Song.songYear = Sings.songYear "
				+ "and Song.studioName = Studio.studioName "
				+ "and SongSinger.starName not in ("
				+ "select SongSinger.starName from SongSinger, Studio, Sings, Song "
				+ "where SongSinger.address <> Studio.address "
				+ "and SongSinger.starName = Sings.starName "
				+ "and Song.songTitle = Sings.songTitle "
				+ "and Song.songYear = Sings.songYear "
				+ "and Song.studioName = Studio.studioName)";
		System.out.println(query);
		ResultSet result = statement.executeQuery(query);
		System.out.println("Forth Query Result:");
		printResult(result);
	}
	
	public void Q5() throws SQLException {
		String query = "select starName	from SongSinger "
				+ "where starName not in ("
				+ "select starName from SongSinger, Song "
				+ "where studioName = 'Sony' "
				+ "and starName not in ("
				+ "select starName from Sings, Song "
				+ "where Sings.songTitle = Song.songTitle "
				+ "and Sings.songYear = Song.songYear "
				+ "and Song.studioName = 'Sony'))";
			System.out.println(query);
			ResultSet result = statement.executeQuery(query);
			System.out.println("Fifth Query Result:");
			printResult(result);
		}
	
	public DBMS() throws SQLException {
		connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/Project Assignment", "root", "password");
		statement = connection.createStatement();
		sc = new Scanner(System.in);
	}
 
	public static void main(String[] args) throws SQLException {
		DBMS system = new DBMS();
		system.Q1();
		system.Q2();
		system.Q3();
		system.Q4();
		system.Q5();
		system.sc.close();
	}
}