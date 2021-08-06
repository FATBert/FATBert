package com.github.othonasgkavardinas.app;

import java.util.ArrayList;
import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;

public class FileWriter extends FileIO{
	private PrintWriter outputWriter;
	
	public FileWriter(String path){
		super(path);
	}
	
	public boolean init(){
		try{
			FileOutputStream outputStream = new FileOutputStream(super.getPath());		
			outputWriter = new PrintWriter(outputStream);
			return true;
		} catch(FileNotFoundException e){
			return false;
		}
	}
	
	public void writeFile(ArrayList<UniversityScore> array){
		for(int i=0; i<array.size(); i++)
			outputWriter.println(array.get(i));
	}
	
	public void close(){
		outputWriter.close();
	}
}
