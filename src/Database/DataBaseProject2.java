package Database;

import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.border.EmptyBorder;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Calendar;

public  class DataBaseProject2 {
	
	private static Statement stmt;
	
	private JFrame frame = new JFrame("Database Project 2");
	private JPanel holder = new JPanel(new GridBagLayout());
	private JButton skw = new JButton("Search Item By Word");
	private JButton sbpa = new JButton("Search Item Above");
	private JButton sbpb = new JButton("Search Item Blow");
	private JButton sir = new JButton("See Item Recommendation");
	private JButton checkSC = new JButton("Check Items");
	private JButton checkOrder = new JButton("Check order");
	private JTextField customer = new JTextField("Enter Customer ID here");
	
	private int cid = -1;
	private int nextOrder;
	
	public DataBaseProject2() {
		
		skw.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if(cid == -1) {
					JOptionPane.showMessageDialog(frame, "Customer ID Required!", "STOP!",JOptionPane.ERROR_MESSAGE);
					return;
				}
				final JFrame result = new JFrame("Search By Key Word");
				JPanel panel = new JPanel(new GridBagLayout());
				
				final JTextField input = new JTextField();
				JButton search = new JButton("search");
				
				final DefaultListModel<String> list = new DefaultListModel<String>();
  	      	  	final JList<String> items = new JList<String>(list);
  	      	  	final ArrayList<String[]> itemArray = new ArrayList<String[]>();
    	      
				search.addActionListener(new ActionListener() {
					@Override
					public void actionPerformed(ActionEvent e) {
						try {
							ResultSet rs = searchItem(input.getText());
							list.removeAllElements();
							while(itemArray.size() > 0) {
								itemArray.remove(0);
							}
							while(rs.next()) {
								list.addElement(rs.getString(1));
								String[] s = new String[4];
								for (int i = 0; i < 4; i++)
									s[i] = rs.getString(i+1);
								itemArray.add(s);
							}
						} catch (SQLException e1) {e1.printStackTrace();}
					}
				});
				
				items.addMouseListener(new MouseAdapter() {
					@Override
					public void mouseClicked(MouseEvent e) {
						if (items.isSelectionEmpty()) return;
						String[] s = itemArray.get(items.getSelectedIndex());
						JOptionPane.showMessageDialog(result, s[0] + " " + s[1] + " " + s[2] + " " + s[3]);
					}
				});
  	      	  	
  	      	  	JButton add = new JButton("add");
  	      	  	
  	      	  	add.addActionListener(new ActionListener() {
	      	  		@Override
	      	  		public void actionPerformed(ActionEvent e) {
	      	  			if (items.isSelectionEmpty()) {
	      	  				JOptionPane.showMessageDialog(result, "Please Select An Item", "Error",JOptionPane.ERROR_MESSAGE);
	      	  				return;
	      	  			}
	      	  			try {
	      	  				int itemID = Integer.valueOf(itemArray.get(items.getSelectedIndex())[1]);
	      	  				int quantity = Integer.valueOf(JOptionPane.showInputDialog("Input Quantity:"));
	      	  				double price = Double.valueOf(itemArray.get(items.getSelectedIndex())[2]);
	      	  				addToCart(cid, itemID, quantity, price);
	      	  			}
	      	  			catch (NumberFormatException e1) {
	      	  				JOptionPane.showMessageDialog(result, "Enter a valid quantity and try again.", "Error",JOptionPane.ERROR_MESSAGE);
	      	  			}
	      	  			catch (SQLException e1) {e1.printStackTrace();}
	      	  		}
	      	  	});
  	      	  	
  	      	  	GridBagConstraints c = new GridBagConstraints();
  	      	  	
  	      	  	c.gridx = 0;
  	      	  	c.gridy = 0;
  	      	  	c.weightx = 1;
  	      	  	c.weighty = 1;
  	      	  	c.fill = GridBagConstraints.BOTH;
  	      	  	c.gridwidth = 2;
  	      	  	panel.add(input, c);
  	      	  	
  	      	  	c.gridy = 1;
  	      	  	c.weighty = 5;
  	      	  	panel.add(new JScrollPane(items), c);
  	      	  	
  	      	  	c.gridy = 2;
  	      	  	c.weighty = 1;
  	      	  	c.gridwidth = 1;
  	      	  	panel.add(search, c);
  	      	  	
  	      	  	c.gridx = 1;
  	      	  	panel.add(add, c);
  	      	  	
  	      	  	result.add(panel);
  	      	  	result.setSize(500,500);
  	      	  	result.setVisible(true);
			}
		});
    
		sbpa.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if(cid == -1) {
					JOptionPane.showMessageDialog(frame, "Customer ID Required!", "STOP!",JOptionPane.ERROR_MESSAGE);
					return;
				}
				final JFrame result = new JFrame("Search Items By Price Above a Threshold");
				JPanel panel = new JPanel(new GridBagLayout());
				final JTextField input = new JTextField();

				JButton search = new JButton("search");
				
				final DefaultListModel<String> list = new DefaultListModel<String>();
				final JList<String> items = new JList<String>(list);
				final ArrayList<String[]> itemArray = new ArrayList<String[]>();
    	      
				search.addActionListener(new ActionListener() {
      	      		@Override
      	      		public void actionPerformed(ActionEvent e) {
      	      			String price = input.getText();
      	      			try {
      	      				double p = Double.parseDouble(price);
      	      				list.removeAllElements();
      	      				ResultSet rs = searchPriceHigh(p);
      	      				while (itemArray.size() > 0) {
      	      					itemArray.remove(0);
      	      				}
      	      				while (rs.next()) {
      	      					list.addElement(rs.getString(1));
      	      					String[] s = new String[4];
      	      					for (int i = 0; i < 4; i++)
      	      						s[i] = rs.getString(i+1);
      	      					itemArray.add(s);
      	      				}
      	      			}
      	      			catch(NumberFormatException e1){
      	      				JOptionPane.showMessageDialog(result, "Must be Numbers!", "STOP!",JOptionPane.ERROR_MESSAGE);
      	      			}
      	      			catch(SQLException e1) {
      	      				e1.printStackTrace();
      	      			}
      	      		}
				});
				
				items.addMouseListener(new MouseAdapter() {
					@Override
					public void mouseClicked(MouseEvent e) {
						if (items.isSelectionEmpty()) return;
						String[] s = itemArray.get(items.getSelectedIndex());
						JOptionPane.showMessageDialog(result, s[0] + " " + s[1] + " " + s[2] + " " + s[3]);
					}
				});
				
				JButton add = new JButton("add");
  	      	  	
				add.addActionListener(new ActionListener() {
  	      	  		@Override
  	      	  		public void actionPerformed(ActionEvent e) {
  	      	  			if (items.isSelectionEmpty()) {
  	      	  				JOptionPane.showMessageDialog(result, "Please Select An Item", "Error",JOptionPane.ERROR_MESSAGE);
  	      	  				return;
  	      	  			}
  	      	  			try {
  	      	  				int itemID = Integer.valueOf(itemArray.get(items.getSelectedIndex())[1]);
  	      	  				int quantity = Integer.valueOf(JOptionPane.showInputDialog("Input Quantity:"));
  	      	  				double price = Double.valueOf(itemArray.get(items.getSelectedIndex())[2]);
  	      	  				addToCart(cid, itemID, quantity, price);
  	      	  			}
  	      	  			catch (NumberFormatException e1) {
  	      	  				JOptionPane.showMessageDialog(result, "Enter a valid quantity and try again.", "Error",JOptionPane.ERROR_MESSAGE);
  	      	  			}
  	      	  			catch (SQLException e1) {e1.printStackTrace();}
  	      	  		}
  	      	  	});
  	      	  	
  	      	  	GridBagConstraints c = new GridBagConstraints();
  	      	  	
  	      	  	c.gridx = 0;
  	      	  	c.gridy = 0;
  	      	  	c.weightx = 1;
  	      	  	c.weighty = 1;
  	      	  	c.fill = GridBagConstraints.BOTH;
  	      	  	c.gridwidth = 2;
  	      	  	panel.add(input, c);
  	      	  	
  	      	  	c.gridy = 1;
  	      	  	c.weighty = 5;
  	      	  	panel.add(new JScrollPane(items), c);
  	      	  	
  	      	  	c.gridy = 2;
  	      	  	c.weighty = 1;
  	      	  	c.gridwidth = 1;
  	      	  	panel.add(search, c);
  	      	  	
  	      	  	c.gridx = 1;
  	      	  	panel.add(add, c);
  	      	  	
				result.add(panel);
				result.setSize(500,500);
				result.setVisible(true);
			}
    		
		});
    
		sbpb.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if(cid == -1) {
					JOptionPane.showMessageDialog(frame, "Customer ID Required!", "STOP!",JOptionPane.ERROR_MESSAGE);
					return;
				}
				final JFrame result = new JFrame("Search Item By Below a Threshold");
				JPanel panel = new JPanel(new GridBagLayout());
				
				final JTextField input = new JTextField();
				
				JButton search = new JButton("search");
				
				final DefaultListModel<String> list = new DefaultListModel<String>();
				final JList<String> items = new JList<String>(list);
				final ArrayList<String[]> itemArray = new ArrayList<String[]>();
    	      
				search.addActionListener(new ActionListener() {
					@Override
					public void actionPerformed(ActionEvent e) {
						String price = input.getText();
						try {
							double p = Double.parseDouble(price);
							list.removeAllElements();
      	      				ResultSet rs = searchPriceLow(p);
      	      				while (itemArray.size() > 0) {
      	      					itemArray.remove(0);
      	      				}
      	      				while (rs.next()) {
      	      					list.addElement(rs.getString(1));
      	      					String[] s = new String[4];
      	      					for (int i = 0; i < 4; i++)
      	      						s[i] = rs.getString(i+1);
      	      					itemArray.add(s);
      	      				}
						}
						catch(NumberFormatException e1) {
							JOptionPane.showMessageDialog(result, "Must be Numbers!", "STOP!",JOptionPane.ERROR_MESSAGE);
						}
						catch(SQLException e1) {
      	      				e1.printStackTrace();
      	      			}
					}
				});
				
				items.addMouseListener(new MouseAdapter() {
					@Override
					public void mouseClicked(MouseEvent e) {
						if (items.isSelectionEmpty()) return;
						String[] s = itemArray.get(items.getSelectedIndex());
						JOptionPane.showMessageDialog(result, s[0] + " " + s[1] + " " + s[2] + " " + s[3]);
					}
				});
				
				JButton add = new JButton("add");
  	      	  	
				add.addActionListener(new ActionListener() {
  	      	  		@Override
  	      	  		public void actionPerformed(ActionEvent e) {
  	      	  			if (items.isSelectionEmpty()) {
  	      	  				JOptionPane.showMessageDialog(result, "Please Select An Item", "Error",JOptionPane.ERROR_MESSAGE);
  	      	  				return;
  	      	  			}
  	      	  			try {
  	      	  				int itemID = Integer.valueOf(itemArray.get(items.getSelectedIndex())[1]);
  	      	  				int quantity = Integer.valueOf(JOptionPane.showInputDialog("Input Quantity:"));
  	      	  				double price = Double.valueOf(itemArray.get(items.getSelectedIndex())[2]);
  	      	  				addToCart(cid, itemID, quantity, price);
  	      	  			}
  	      	  			catch (NumberFormatException e1) {
  	      	  				JOptionPane.showMessageDialog(result, "Enter a valid quantity and try again.", "Error",JOptionPane.ERROR_MESSAGE);
  	      	  			}
  	      	  			catch (SQLException e1) {e1.printStackTrace();}
  	      	  		}
  	      	  	});
  	      	  	
  	      	  	GridBagConstraints c = new GridBagConstraints();
  	      	  	
  	      	  	c.gridx = 0;
  	      	  	c.gridy = 0;
  	      	  	c.weightx = 1;
  	      	  	c.weighty = 1;
  	      	  	c.fill = GridBagConstraints.BOTH;
  	      	  	c.gridwidth = 2;
  	      	  	panel.add(input, c);
  	      	  	
  	      	  	c.gridy = 1;
  	      	  	c.weighty = 5;
  	      	  	panel.add(new JScrollPane(items), c);
  	      	  	
  	      	  	c.gridy = 2;
  	      	  	c.weighty = 1;
  	      	  	c.gridwidth = 1;
  	      	  	panel.add(search, c);
  	      	  	
  	      	  	c.gridx = 1;
  	      	  	panel.add(add, c);
				
				result.add(panel);
				result.setSize(500,500);
				result.setVisible(true);
			}
		});
		
		sir.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if(cid == -1) {
					JOptionPane.showMessageDialog(frame, "Customer ID Required!", "STOP!",JOptionPane.ERROR_MESSAGE);
					return;
				}
				final JFrame result = new JFrame("Item Recommendation");
				JPanel panel = new JPanel(new GridBagLayout());
				
				final DefaultListModel<String> list = new DefaultListModel<String>();
				final JList<String> items = new JList<String>(list);
				final ArrayList<String[]> itemArray = new ArrayList<String[]>();
				
				try {
					ResultSet rs = recommendItem(cid);
					list.removeAllElements();
					while(itemArray.size() > 0) {
						itemArray.remove(0);
					}
					while(rs.next()) {
						list.addElement(rs.getString(1));
						String[] s = new String[4];
						for (int i = 0; i < 4; i++)
							s[i] = rs.getString(i+1);
						itemArray.add(s);
					}
				} catch (SQLException e1) {e1.printStackTrace();}
				
				items.addMouseListener(new MouseAdapter() {
					@Override
					public void mouseClicked(MouseEvent e) {
						if (items.isSelectionEmpty()) return;
						String[] s = itemArray.get(items.getSelectedIndex());
						JOptionPane.showMessageDialog(result, s[0] + " " + s[1] + " " + s[2] + " " + s[3]);
					}
				});
				
				JButton add = new JButton("add");
  	      	  	
  	      	  	add.addActionListener(new ActionListener() {
  	      	  		@Override
  	      	  		public void actionPerformed(ActionEvent e) {
  	      	  			if (items.isSelectionEmpty()) {
  	      	  				JOptionPane.showMessageDialog(result, "Please Select An Item", "Error",JOptionPane.ERROR_MESSAGE);
  	      	  				return;
  	      	  			}
  	      	  			try {
  	      	  				int itemID = Integer.valueOf(itemArray.get(items.getSelectedIndex())[1]);
  	      	  				int quantity = Integer.valueOf(JOptionPane.showInputDialog("Input Quantity:"));
  	      	  				double price = Double.valueOf(itemArray.get(items.getSelectedIndex())[2]);
  	      	  				addToCart(cid, itemID, quantity, price);
  	      	  			}
  	      	  			catch (NumberFormatException e1) {
  	      	  				JOptionPane.showMessageDialog(result, "Enter a valid quantity and try again.", "Error",JOptionPane.ERROR_MESSAGE);
  	      	  			}
  	      	  			catch (SQLException e1) {e1.printStackTrace();}
  	      	  		}
  	      	  	});
  	      	  	
  	      	  	GridBagConstraints c = new GridBagConstraints();
  	      	  	
  	      	  	c.gridx = 0;
  	      	  	c.gridy = 0;
  	      	  	c.weightx = 1;
  	      	  	c.weighty = 5;
  	      	  	c.fill = GridBagConstraints.BOTH;
  	      	  	panel.add(new JScrollPane(items), c);
  	      	  	
  	      	  	c.gridy = 2;
  	      	  	c.weighty = 1;
  	      	  	panel.add(add, c);
				
				result.add(panel);
				result.setSize(500,500);
				result.setVisible(true);
			}
		});
    
		checkSC.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if(cid == -1){
					JOptionPane.showMessageDialog(frame, "Customer ID Required!", "STOP!",JOptionPane.ERROR_MESSAGE);
					return;
				}
				final JFrame result = new JFrame("All Items Currently in Shopping Cart");
				JPanel panel = new JPanel(new GridBagLayout());

				final DefaultListModel<String> list = new DefaultListModel<String>();
				final ArrayList<String[]> listItems = new ArrayList<String[]>();
				
				try {
					ResultSet rs = viewCart(cid);
					while(rs.next()) {
						list.addElement(rs.getString(4) + "  *  " + rs.getString(2) + "  Price: " + rs.getDouble(3));
						String[] s = new String[2];
						s[0] = rs.getString(1);
						s[1] = rs.getString(2);
						listItems.add(s);
					}
				} catch (SQLException e1) {e1.printStackTrace();}
				
				final JList<String> sc = new JList<String>(list);
				
				JButton remove = new JButton("Remove");
				remove.addActionListener(new ActionListener() {
					@Override
					public void actionPerformed(ActionEvent e) {
						if (sc.isSelectionEmpty()) {
  	      	  				JOptionPane.showMessageDialog(result, "Select An Item", "Error",JOptionPane.ERROR_MESSAGE);
  	      	  				return;
  	      	  			}
						
						try {
							int itemID = Integer.valueOf(listItems.get(sc.getSelectedIndex())[0]);
							deleteFromCart(itemID, cid);
						} catch (NumberFormatException e1) {
							e1.printStackTrace();
						} catch (SQLException e1) {
							e1.printStackTrace();
						}
						
						listItems.remove(sc.getSelectedIndex());
						list.remove(sc.getSelectedIndex());
						
					}
				});
				
				JButton checkout = new JButton("Checkout");
				checkout.addActionListener(new ActionListener() {
					@Override
					public void actionPerformed(ActionEvent e) {
						if(list.isEmpty()) {
							JOptionPane.showMessageDialog(result, "Cannot check out an empty shopping cart.", "Error",JOptionPane.ERROR_MESSAGE);
							return;
						}
						try {
							placeOrder(nextOrder, cid);
						} catch (SQLException e1) {
							e1.printStackTrace();
						}
						for(int i = 0; i < list.size(); i++) {
							int itemID = Integer.valueOf(listItems.get(0)[0]);
							int quantity = Integer.valueOf(listItems.get(0)[1]);
							try {
								placeOrder2(nextOrder, itemID, quantity);
								deleteFromCart(itemID, cid);
								
							} catch (SQLException e1) {
								e1.printStackTrace();
							}
							listItems.remove(0);
						}
						list.removeAllElements();
						nextOrder++;
					}
				});
				
				GridBagConstraints c = new GridBagConstraints();
  	      	  	
  	      	  	c.gridx = 0;
  	      	  	c.gridy = 0;
  	      	  	c.weightx = 1;
  	      	  	c.weighty = 5;
  	      	  	c.fill = GridBagConstraints.BOTH;
  	      	  	c.gridwidth = 2;
  	      	  	panel.add(new JScrollPane(sc), c);
  	      	  	
  	      	  	c.gridy = 1;
  	      	  	c.weighty = 1;
  	      	  	c.gridwidth = 1;
  	      	  	panel.add(remove, c);
  	      	  	
  	      	  	c.gridx = 1;
  	      	  	panel.add(checkout, c);
				
  	      	  	result.add(panel);
				result.setSize(500,500);
				result.setVisible(true);
			}
		});
    
		checkOrder.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if(cid == -1){
					JOptionPane.showMessageDialog(frame, "Customer ID Required!", "STOP!",JOptionPane.ERROR_MESSAGE);
					return;
				}
				final JFrame result = new JFrame("All Orders");
				
				final DefaultListModel<Integer> list = new DefaultListModel<Integer>();
				final JList<Integer> orders = new JList<Integer>(list);
				final ArrayList<String> dates = new ArrayList<String>();
				
				try {
					ResultSet rs = viewOrderCustomer(cid);
					while(rs.next()) {
						list.addElement(rs.getInt(1));
						dates.add(rs.getString(2));
					}
				} catch (SQLException e1) {e1.printStackTrace();}
				
				orders.addMouseListener(new MouseAdapter() {
					@Override
					public void mouseClicked(MouseEvent arg0) {
						if(cid == -1){
							JOptionPane.showMessageDialog(result, "Customer ID Required!", "STOP!",JOptionPane.ERROR_MESSAGE);
							return;
						}
						String title = orders.getSelectedValue().toString() + " " + dates.get(orders.getSelectedIndex());
						final JFrame details = new JFrame(title);
						
						final DefaultListModel<String> list = new DefaultListModel<String>();
						final JList<String> items = new JList<String>(list);
						final ArrayList<String[]> itemDetails = new ArrayList<String[]>();
						
						try {
							ResultSet rs = viewOrderCustomer2(cid, orders.getSelectedValue());
							while (rs.next()) {
								list.addElement(rs.getString(1));
								String[] s = new String[3];
								s[0] = rs.getString(1);
								s[1] = rs.getString(2);
								s[2] = rs.getString(4);
								itemDetails.add(s);
							}
						} catch (SQLException e1) {e1.printStackTrace();}
						
						items.addMouseListener(new MouseAdapter() {
							@Override
							public void mouseClicked(MouseEvent e) {
								int i = items.getSelectedIndex();
								String s = "Name: " + itemDetails.get(i)[0] +
										"\nQuantity: " + itemDetails.get(i)[1] + 
										"\nPrice per Item: " + itemDetails.get(i)[2] +
										"\nTotal Spending: " + 
										(Double.valueOf(itemDetails.get(i)[2]) * Integer.valueOf(itemDetails.get(i)[1]));
								JOptionPane.showMessageDialog(details, s);
							}
						});
						
						details.add(new JScrollPane(items));
						details.setSize(500,500);
						details.setVisible(true);
					}
				});
				
				result.add(new JScrollPane(orders));
				result.setSize(500,500);
				result.setVisible(true);
			}
		});
    
		customer.addKeyListener(new KeyListener() {

			@Override
			public void keyPressed(KeyEvent arg0) {
			}

			@Override
			public void keyReleased(KeyEvent arg0) {
				try {
					cid = Integer.valueOf(customer.getText());
				}
				catch (NumberFormatException e) {
					cid = -1; return;
				}
			}
			
			@Override
			public void keyTyped(KeyEvent arg0) {}
    	
		});
		
		try {
			nextOrder = maxOrder() + 1;
		} catch (SQLException e1) {e1.printStackTrace();}
		
		GridBagConstraints c = new GridBagConstraints();
		frame.setSize(900,600);
		holder.setBorder(new EmptyBorder(50, 50, 50, 50));
		holder.setBackground(Color.cyan);
    
		c.gridx = 0;
		c.gridy = 0;
		c.fill = GridBagConstraints.BOTH;
		c.anchor = GridBagConstraints.NORTH;
		c.gridwidth = 4;
		c.weightx = 1;
		c.weighty = 4;
		holder.add(customer,c);
    
		c.gridy = 1;
		c.gridwidth = 1;
		holder.add(skw,c);
    
		c.gridx = 1;
		holder.add(sbpa,c);
    
		c.gridx = 2;
		holder.add(sbpb,c);
		
		c.gridx = 3;
		holder.add(sir,c);
    
		c.gridy = 2;
		c.gridx = 0;
		c.gridwidth = 2;
		holder.add(checkSC,c);
    
		c.gridx = 2;
		holder.add(checkOrder,c);
    
		frame.add(holder);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
	
	public static ResultSet searchItem(String in) throws SQLException{
		String query = "select \"Item\".\"ItemName\",\"Item\".\"ItemID\",\"Item\".\"Price\",\"Item\".\"ItemDescription\" "
				+ "from \"Item\" "
				+ "where \"ItemName\" LIKE '%" + in +"%' ";
		return stmt.executeQuery(query);
	}
	
	public static ResultSet viewCart(int customerID) throws SQLException{
		String query = "select \"Shopping_Cart\".\"ItemID\", \"Shopping_Cart\".\"Quantity\", sum(\"Shopping_Cart\".\"TotalPrice\"),"
				+ "\"Item\".\"ItemName\""
				+ " from \"Shopping_Cart\", \"Item\" "
				+ "Where \"Shopping_Cart\".\"CustomerID\" = "+ customerID 
				+"and \"Item\".\"ItemID\" = \"Shopping_Cart\".\"ItemID\""
				+ "group by \"ItemName\", \"Shopping_Cart\".\"ItemID\", \"Shopping_Cart\".\"Quantity\"";
		
		return stmt.executeQuery(query);
	}
	
	public static ResultSet searchPriceHigh(double price) throws SQLException{
		String query = "select \"Item\".\"ItemName\",\"Item\".\"ItemID\",\"Item\".\"Price\",\"Item\".\"ItemDescription\" "
				+ "from \"Item\" "
				+ "where \"Price\" >= " + price;
		return stmt.executeQuery(query);
	}
	
	public static ResultSet searchPriceLow(double price) throws SQLException{
		String query = "select \"Item\".\"ItemName\",\"Item\".\"ItemID\",\"Item\".\"Price\",\"Item\".\"ItemDescription\" "
				+ "from \"Item\" "
				+ "where \"Price\" <= " + price;
						//+ "group by \"ItemName\"";
		return stmt.executeQuery(query);
	}

	public static int searchItemInCart(int ItemID, int customerID) throws SQLException{
		String query = "select \"Shopping_Cart\".\"Quantity\" "
				+ "from \"Shopping_Cart\" "
				+ "where \"Shopping_Cart\".\"ItemID\" = "+ ItemID
						+"and \"Shopping_Cart\".\"CustomerID\" = "+ customerID;
		ResultSet rs = stmt.executeQuery(query);
		String Output = "";
		if (rs.next()) {
			Output = rs.getString(1);
		}
		try {
			return Integer.valueOf(Output);
		}
		catch (NumberFormatException e) {
			return 0;
		}
	}
	
	public static void addToCart(int CustomerID, int ItemID, int Quantity, double price) throws SQLException{
		int quantity = searchItemInCart(ItemID, CustomerID);
		if(quantity == 0){
			String query = "insert INTO \"Shopping_Cart\" (\"CustomerID\",\"ItemID\",\"Quantity\",\"TotalPrice\") VALUES ( "
					+ CustomerID+","+ItemID+","+ Quantity+","+ Quantity * price+")";
			stmt.executeUpdate(query);
		}
		else{
			int total = quantity + Quantity;
			double totalPrice = total * price;
			String query = "update \"Shopping_Cart\""
					+ "set \"Quantity\" = "+ total +","
					+ "\"TotalPrice\" = " + totalPrice +
					"where \"Shopping_Cart\".\"ItemID\" = "+ ItemID +
					"and \"Shopping_Cart\".\"CustomerID\" = "+ CustomerID;
			stmt.executeUpdate(query);
		}
	}
	
	public static void deleteFromCart(int ItemID ,int customerID) throws SQLException{
		String query = "delete from \"Shopping_Cart\" "
				+"where \"Shopping_Cart\".\"ItemID\" = "+ ItemID +
				"and \"Shopping_Cart\".\"CustomerID\" = "+ customerID;
		stmt.executeUpdate(query);
	}
	
	public static void placeOrder(int orderID, int CustomerID) throws SQLException{
		Calendar c = Calendar.getInstance();
		int mo = c.get(Calendar.MONTH);
		int da = c.get(Calendar.DATE);
		int ya = c.get(Calendar.YEAR);
        String query = "insert INTO \"Orders\" (\"OrderID\", \"CustomerID\", \"Date\")"
        		+ "VALUES ("+ orderID +","
        		+ CustomerID+","
        		+ "TO_DATE('"+ mo +"/" + da + "/" + ya +"', 'mm/dd/yyyy')"
        		+")";
        stmt.executeUpdate(query);
	}
	
	public static void placeOrder2(int orderID, int ItemID, int Quantity) throws SQLException {
		 String query = "insert into \"Sales\""
	        		+ "(\"OrderID\",\"ItemID\",\"Quantity\")"
	        		+ "VALUES ( "+ orderID + ","+ ItemID +","+ Quantity + " )";
		 stmt.executeUpdate(query);
	}
	
	public static ResultSet viewOrderCustomer(int customerID) throws SQLException{
		String query = "select \"Orders\".\"OrderID\",\"Orders\".\"Date\""
				+ "from \"Orders\" "
				+ "Where \"Orders\".\"CustomerID\" = "+ customerID ;
		
		return stmt.executeQuery(query);
	}
	
	public static ResultSet viewOrderCustomer2(int customerID ,int OrderID) throws SQLException{
		String query = "select distinct \"Item\".\"ItemName\", \"Sales\".\"Quantity\" ,\"Item\".\"ItemID\",\"Item\".\"Price\" "
				+ "from \"Orders\", \"Item\",\"Sales\" "
				+ "Where \"Orders\".\"CustomerID\" = "+ customerID 
				+"and \"Sales\".\"OrderID\" =" + OrderID
				+"and \"Sales\".\"ItemID\" = \"Item\".\"ItemID\"";
		
		return stmt.executeQuery(query);
	}
	
	public static ResultSet recommendItem(int CustomerID) throws SQLException{
		String query = "select distinct \"Item\".\"ItemName\",\"Item\".\"ItemID\",\"Item\".\"Price\",\"Item\".\"ItemDescription\" "
				+ "from  \"Sales\", \"Orders\",\"Item\" "
				+ "where \"Sales\".\"OrderID\" = \"Orders\".\"OrderID\""
				+ "and \"Item\".\"ItemID\"= \"Sales\".\"ItemID\" "
				+ "and \"Orders\".\"CustomerID\" = " + CustomerID
				+ "and not exists( select \"Sales\" .\"ItemID\" "
				+ " from \"Sales\", \"Orders\" "
				+ "where \"Sales\".\"OrderID\" = \"Orders\".\"OrderID\" "
				+ "and \"Item\".\"ItemID\"= \"Sales\".\"ItemID\" "
				+ "and \"Orders\".\"CustomerID\" <> " + CustomerID +")";
		
		return stmt.executeQuery(query);
	}
	
	public int maxOrder() throws SQLException {
		String query = "select MAX(\"Orders\".\"OrderID\") from \"Orders\" ";
		ResultSet rs = stmt.executeQuery(query);
		if (rs.next()) {
			return rs.getInt(1);
		}
		else return 0;
	}
  
	public static void main(String args[]) {
		
		try {
			Class.forName("org.postgresql.Driver");
			Connection con = DriverManager.getConnection("jdbc:postgresql://localhost/EECS341_Project_Part2", "postgres", "123456");  
			stmt = con.createStatement();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (SQLException e) {
			e.printStackTrace();
		}  
		
		new DataBaseProject2();
	}
}