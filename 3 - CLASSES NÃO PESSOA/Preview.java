//Generate docx 出货单 for preview

package javaGui;


import com.spire.doc.*;
import com.spire.doc.documents.*;
//import com.spire.doc.fields.DocPicture;
import com.spire.doc.fields.TextRange;


import com.spire.doc.FieldType;
import com.spire.doc.FileFormat;
import com.spire.doc.HeaderFooter;
import com.spire.doc.documents.HorizontalAlignment;



import com.spire.doc.documents.Paragraph;
import com.spire.doc.documents.ParagraphStyle;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.awt.*;
import java.util.ArrayList;

public class Preview {
	
	   public static void main(String[] args){

	        String [] custom = {"0.0077","name","addr","city","state","zip","Phone"};
	        String goods[][]  = {
	        		new String[]{"11","bag","Taita", "22.00", "Yes", "太太基金"},
	        		new String[]{"2","cs","space X", "122.24", "No", "出前一丁"},
	        		};
	        top( custom,goods);
	   }
	   
	   
	   
	   
	   public static void top(String[] custom,String [][]goods){      
	        
	        String [][]data1 = genadd(custom);

	        String [][]data2 = genitem(goods);
	        
	        String [][]data3 = gendig(goods,custom);
	        
	        lowerLevel(data1, data2,data3);
	   }
	   
	   public static String[][] gendig(String [][] In, String []Input)
	   {
		   String tt[] = gentax( In, Input);

           String res[][] = {
               new String[]{"					","Subtotal: ",  "$"+tt[1]},
               new String[]{"					","Tax: ", 		"$"+tt[2]},
               new String[]{"Number of Items: "+ tt[0],"Total: ", "$"+tt[3] },
               new String[]{" "," ", " " },
       		};	     
           return res;
	   }
	   
	   
	   public static String getSysTime()
	   {
	        SimpleDateFormat sdf = new SimpleDateFormat();// 格式化时间 
	        
	        sdf.applyPattern("MM/dd/yyyy");
	        
//	        sdf.applyPattern("yyyy/MM/dd");// a为am/pm的标记  
	        Date date = new Date();// 获取当前时间 
	        
	        String shijian  = sdf.format(date);
	        System.out.println(shijian); 
	        return shijian;
	   }
	   
	   
	   public static String[] gentax(String [][] In, String []Input)
	   {
		   
		   java.text.DecimalFormat myformat=new java.text.DecimalFormat("#0.00");
		   int size = In.length;
		   double sum= 0;
		   double tax= 0;
		   for(int i = 0;i<size;i++)
		   {
		
			   int qty = Integer.valueOf(In[i][0]).intValue();
	
			   double price = Double.parseDouble(In[i][3]);//装换为double类型
			   
			   price=Double.parseDouble(myformat.format(price));//保留bai2为小数
			   double num = qty* price;

			   
			   if(In[i][4]=="Yes")
			   {
				   double taxrate = Double.parseDouble(Input[0]);
				   tax +=  Double.parseDouble(myformat.format(num*qty*taxrate));
			   }

			   
			   num=Double.parseDouble(myformat.format(num));
			   sum+=num;
		   }
		   
		   tax = Double.parseDouble(myformat.format(tax));
		   String Num = Integer.toString(size);
		   String Subtotal = Double.toString(sum);
		   String TAX = Double.toString(tax);
		   String Total = Double.toString(sum + tax);
		   
		   String[] res =  {Num,Subtotal,TAX,Total};
		   return res;
	   }
	   	   	   
	   public static String[][] genitem(String [][] In )
	   {
		   
		   ArrayList<String[]> arr = new ArrayList<>();
		   int size = In.length;
		   for(int i = 0;i<size;i++)
		   {
		
			   int qty = Integer.valueOf(In[i][0]).intValue();
			   
			   double price = Double.parseDouble(In[i][3]);//装换为double类型
			   java.text.DecimalFormat myformat=new java.text.DecimalFormat("#0.00");
			   price=Double.parseDouble(myformat.format(price));//保留bai2为小数
			   
			   double num = qty* price;
			   num=Double.parseDouble(myformat.format(num));
			   
			   String[] a = {In[i][0]+" "+In[i][1], In[i][2]+" "+In[i][5],"$"+In[i][3], "","$"+ Double.toString(num)} ;
			   arr.add(a);
		   }
		   
		   return (String[][])arr.toArray(new String[0][]);
	   }
	   	   
	   
	   
	   
	   public static String[][] genadd(String [] input )  // add Invoice Number here...
	   {
		   String sysTime = getSysTime();
	        
		   String[][] res =
			   {
					 new String[]{"Sold To:",""},   
					 new String[]{"    "+input[1],"Invoice Number: "},
					 new String[]{"    "+input[2],"Date: "+ sysTime},
					 new String[]{"    "+input[3]+" ,"+input[4]+ " " + input[5] ,"P.O.Number:"},
					 new String[]{"    Tel: "+input[6] ,"Terms: 30days"}
			   };
		   return res;
	   }
	   
	   
	   
	   public static void lowerLevel( String [][]data2 ,String [][]data ,String [][]data3)
	   {
	        //创建Document对象
		   	String textstyle = "Times New Roman";
	        Document doc = new Document();
	        Section sec = doc.addSection();
	        Paragraph para1 = sec.addParagraph();
	        para1.appendText("Yee Chong Hon Co.");

	       
	        Paragraph para5 = sec.addParagraph();
	        Paragraph para4 = sec.addParagraph();
	        para5.appendText("Wholesaler & Distributor");
	        para4.appendText("(775) 409-4002           (775) 409-4003");

	        Paragraph para2 = sec.addParagraph();
	        para2.appendText("FAX: (775) 409-4005\n");
	        

	        
	        
	        ParagraphStyle style1 = new ParagraphStyle(doc);
	        style1.setName("titleStyle");
	        style1.getCharacterFormat().setBold(true);
	        style1.getCharacterFormat().setTextColor(Color.BLACK);
	        style1.getCharacterFormat().setFontName(textstyle);
	        style1.getCharacterFormat().setFontSize(24f);
	        doc.getStyles().add(style1);
	        para1.applyStyle("titleStyle");
	        
	        para1.getFormat().setHorizontalAlignment(HorizontalAlignment.Center);
	        para1.getFormat().setAfterSpacing(2f);

	        ParagraphStyle style2 = new ParagraphStyle(doc);
	        style2.setName("bigphone");
	        style2.getCharacterFormat().setBold(true);
	        style2.getCharacterFormat().setTextColor(Color.BLACK);
	        style2.getCharacterFormat().setFontName(textstyle);
	        style2.getCharacterFormat().setFontSize(14f);
	        doc.getStyles().add(style2);
	        para2.applyStyle("bigphone");
	        
	        para2.getFormat().setHorizontalAlignment(HorizontalAlignment.Center);
	        para2.getFormat().setAfterSpacing(20f);	        
	        
	        ParagraphStyle style3 = new ParagraphStyle(doc);
	        style3.setName("soldto");
	        style3.getCharacterFormat().setBold(true);
	        style3.getCharacterFormat().setTextColor(Color.BLACK);
	        style3.getCharacterFormat().setFontName(textstyle);
	        style3.getCharacterFormat().setFontSize(10f);
	        doc.getStyles().add(style3);
//	        

	        
	        ParagraphStyle style4 = new ParagraphStyle(doc);
	        style4.setName("smallphone");
//	        style4.getCharacterFormat().setBold(true);
	        style4.getCharacterFormat().setTextColor(Color.BLACK);
	        style4.getCharacterFormat().setFontName(textstyle);
	        style4.getCharacterFormat().setFontSize(12f);
	        doc.getStyles().add(style4);
	        para4.applyStyle("smallphone");

	        
	        ParagraphStyle style5 = new ParagraphStyle(doc);
	        style5.setName("whole");
//	        style4.getCharacterFormat().setBold(true);
	        style5.getCharacterFormat().setTextColor(Color.BLACK);
	        style5.getCharacterFormat().setFontName(textstyle);
	        style5.getCharacterFormat().setFontSize(10f);
	        doc.getStyles().add(style5);
	        para5.applyStyle("whole");
	       
	        para2.getFormat().setHorizontalAlignment(HorizontalAlignment.Center);
	        para2.getFormat().setAfterSpacing(2f);	        
	        
	        para5.getFormat().setHorizontalAlignment(HorizontalAlignment.Center);
	        para5.getFormat().setAfterSpacing(2f);	  
	        
	        para4.getFormat().setHorizontalAlignment(HorizontalAlignment.Center);
	        para4.getFormat().setAfterSpacing(2f);	
	        
//	        Paragraph para6 = sec.addParagraph();
//	        para6.applyStyle("whole");
//	        
//	        para6.appendText("Mailing Address \t\t\t warehouse\n" 
//	        		+ "P.O.Box 21090                    115E. Moana Lane\n"
//	        		+ "Reno,NV 89515					Reno NV89502");   
//	        
//	        para6.getFormat().setHorizontalAlignment(HorizontalAlignment.Left);
//	        para6.getFormat().setAfterSpacing(12f);	        
	        //声明数组内容
	        String[][] data4 =
	        	{
	        		      new String[]{"Mailing Address\nP.O.Box 21090\nReno,NV 89515",
                  		"Warehouse\n115E. Moana Lane\nReno NV 89502"},

	        	};
	        Table table4 = sec.addTable(true);
	        table4.resetCells(data4.length, data4[0].length);
	        TableRow row = table4.getRows().get(0);
//	        row.setHeight(80);
	        //写入剩余组内容到表格，并格式化数据
	        for (int r = 0; r < data4.length; r++) 
	        {
	            TableRow dataRow = table4.getRows().get(r );
	            dataRow.setHeight(35);
	            dataRow.setHeightType(TableRowHeightType.Exactly);
//	            dataRow.getRowFormat().setBackColor(Color.white);
	            for (int c = 0; c < data4[r].length; c++) 
	            {

	            	dataRow.getCells().get(c).getCellFormat().setVerticalAlignment(VerticalAlignment.Middle);

	                TextRange range2 = dataRow.getCells().get(c).addParagraph().appendText(data4[r][c]);
	                range2.getCharacterFormat().setFontName(textstyle);

	            	range2.getCharacterFormat().setFontSize(9f); 
	                if(c == data4[0].length-1)
	                	 range2.getOwnerParagraph().getFormat().setHorizontalAlignment(HorizontalAlignment.Right);
	                else
	                {
	                	range2.getOwnerParagraph().getFormat().setHorizontalAlignment(HorizontalAlignment.Left);
	            		
	                }
	    	        		
                }
	             
            }        

	        Paragraph para6 = sec.addParagraph();
	        para6.appendText("");
	        


	        //添加表格
	        Table table2 = sec.addTable(true);
	        table2.resetCells(data2.length, data2[0].length);
	        table2.getRows().get(0).getCells().get(0).setWidth(800);
	        table2.getRows().get(0).getCells().get(1).setWidth(400);
	        row = table2.getRows().get(0);
//	        row.setHeight(5);
	        //写入剩余组内容到表格，并格式化数据
	        for (int r = 0; r < data2.length; r++) {
	            TableRow dataRow = table2.getRows().get(r );
	            dataRow.setHeight(14);
	            dataRow.setHeightType(TableRowHeightType.Exactly);
//	            dataRow.getRowFormat().setBackColor(Color.white);
	            for (int c = 0; c < data2[r].length; c++) 
	            {

	            	dataRow.getCells().get(c).getCellFormat().setVerticalAlignment(VerticalAlignment.Middle);

	                TextRange range2 = dataRow.getCells().get(c).addParagraph().appendText(data2[r][c]);
	                range2.getCharacterFormat().setFontName(textstyle);

                   	range2.getCharacterFormat().setFontSize(11f);
                	range2.getCharacterFormat().setBold(true);
      
       
	                	range2.getOwnerParagraph().getFormat().setHorizontalAlignment(HorizontalAlignment.Left);
	 
    
                }
	             
            }
    
	        Paragraph para7 = sec.addParagraph();
	        para7.appendText("");    
	        

	        	        
	        //声明数组内容
	        String[] header = {"Qty","Description","Price", "", "Total"};

	        //添加表格
	        Table table = sec.addTable(true);
	        table.resetCells(data.length + 1, header.length);

	        //设置表格第一行作为表头，写入表头数组内容，并格式化表头数据
	        row = table.getRows().get(0);
//	        TableRow row = table.getRows().get(0);
	        table.getRows().get(0).getCells().get(0).setWidth(120);
	        table.getRows().get(0).getCells().get(1).setWidth(900);
	        row.isHeader(true);
//	        row.setHeight(20);
//	        row.setHeightType(TableRowHeightType.Exactly);
//	        row.getRowFormat().setBackColor(Color.ORANGE);
	        for (int i = 0; i < header.length; i++) {
	            row.getCells().get(i).getCellFormat().setVerticalAlignment(VerticalAlignment.Middle);
	            Paragraph p = row.getCells().get(i).addParagraph();
	            p.getFormat().setHorizontalAlignment(HorizontalAlignment.Center);
	            TextRange range1 = p.appendText(header[i]);
	            range1.getCharacterFormat().setFontName(textstyle);
	            range1.getCharacterFormat().setFontSize(11f);
	            range1.getCharacterFormat().setBold(true);
//	            range1.getCharacterFormat().setTextColor(Color.white);
	        }

	        //写入剩余组内容到表格，并格式化数据
	        for (int r = 0; r < data.length; r++) {
	            TableRow dataRow = table.getRows().get(r + 1);
	            dataRow.getCells().get(0).setWidth(120);
	            dataRow.getCells().get(1).setWidth(900);
//	            dataRow.setHeight(25);
//	            dataRow.setHeightType(TableRowHeightType.Exactly);
//	            dataRow.getRowFormat().setBackColor(Color.white);
	            for (int c = 0; c < data[r].length; c++) {
	
            		dataRow.getCells().get(c).getCellFormat().setVerticalAlignment(VerticalAlignment.Middle);
	                TextRange range2 = dataRow.getCells().get(c).addParagraph().appendText(data[r][c]);
	                
	                range2.getOwnerParagraph().getFormat().setHorizontalAlignment(HorizontalAlignment.Left);
	                range2.getCharacterFormat().setFontName(textstyle);


	   
	                range2.getCharacterFormat().setFontSize(11f);
	
	            }
	        }


	        Paragraph para3 = sec.addParagraph();
	        para3.appendText("\n\n\n\n");



	        //添加表格
	        Table table3 = sec.addTable(true);
	        table3.resetCells(data3.length, data3[0].length);
	        table3.getRows().get(0).getCells().get(0).setWidth(660);
	        row = table3.getRows().get(0);
	        row.setHeight(30);
	        //写入剩余组内容到表格，并格式化数据
	        for (int r = 0; r < data3.length; r++) {
	            TableRow dataRow = table3.getRows().get(r);
	            dataRow.setHeight(16);
	            dataRow.setHeightType(TableRowHeightType.Exactly);
	            dataRow.getRowFormat().setBackColor(Color.white);
	            
	            for (int c = 0; c < data3[r].length; c++) {
	            	
	            	dataRow.getCells().get(c).getCellFormat().setVerticalAlignment(VerticalAlignment.Middle);
	                TextRange range2 = dataRow.getCells().get(c).addParagraph().appendText(data3[r][c]);
	                if(c==0)
	                range2.getOwnerParagraph().getFormat().setHorizontalAlignment(HorizontalAlignment.Left);
	                else if(c>0)
	                	range2.getOwnerParagraph().getFormat().setHorizontalAlignment(HorizontalAlignment.Right);
	                range2.getCharacterFormat().setFontName(textstyle);

	            }
	        }
	        
	        //设置表格边框样式
	        table.getTableFormat().getBorders().setBorderType(BorderStyle.Single);
//	        table.autoFit(AutoFitBehaviorType.Auto_Fit_To_Contents);
	        table2.getTableFormat().getBorders().setBorderType(BorderStyle.None);
	        table4.getTableFormat().getBorders().setBorderType(BorderStyle.None);
//	        table2.autoFit(AutoFitBehaviorType.Fixed_Column_Widths);
//	        table2.autoFit(AutoFitBehaviorType.Auto_Fit_To_Contents);
	        table3.getTableFormat().getBorders().setBorderType(BorderStyle.None);
	        //保存文档
	        doc.saveToFile("./img/Preview.docx", FileFormat.Docx_2013);
	        System.out.println("save done");
	    }

}

