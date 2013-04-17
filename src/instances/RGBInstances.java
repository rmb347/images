

package instances;

import structure.Sequence;
import structure.Sequences;

import image.CONSTANT;
import image.ImageUtils;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.analysis.algorithm.EdgeDirectionCoherenceVector;
import org.openimaj.image.colour.Transforms;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.processing.face.detection.FaceDetector;
import org.openimaj.image.processing.face.detection.keypoints.FKEFaceDetector;
import org.openimaj.image.processing.face.detection.keypoints.FacialKeypoint;
import org.openimaj.image.processing.face.detection.keypoints.KEDetectedFace;
import org.openimaj.math.geometry.shape.Rectangle;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

public class RGBInstances {
	
	int nf;
	
	Instances header;
	HashMap<String,String> url2access = new HashMap<String, String>();
	
	public RGBInstances(){
	}
	
	public void setNF(int nf){
		this.nf = nf;
	}
	
	public List<String> getStaticURL(File filename)
	{
		List<String> URls = new ArrayList<String>();
		if (filename.exists()) {
			try {
				FileInputStream inputStream = new FileInputStream(filename);
				DataInputStream in = new DataInputStream(inputStream);
				BufferedReader br = new BufferedReader(
						new InputStreamReader(in));
				String line = null;
				while ((line = br.readLine()) != null) {
					String[] splitline = line.split(",");
					URls.add(splitline[1]);
					String access = getAccess(splitline[2]);
					url2access.put(splitline[1], access);
				}
				in.close();
				br.close();
			}

			catch (IOException e) {
				e.printStackTrace();
			}
		}
		return URls;
	}
	
	private String getAccess(String string) {
		if ( string.contains("private"))
		{
			return "private";
		}
		
		return "public";
	}

	public void setHeader(){
		
		FastVector attrInfo = new FastVector();
		
		FastVector classVals = new FastVector();
		classVals.addElement("+1");
		classVals.addElement("-1");
		Attribute classAttr = new Attribute("class", classVals);
		
		for (int i = 0; i < nf; i ++){
			Attribute a = new Attribute(new String("a" + i));
			attrInfo.addElement(a);
		}		
		attrInfo.addElement(classAttr);
		
		header = new Instances("data", attrInfo, 0);
		header.setClassIndex(nf);
	}
	
	public Instances getHeader(){
		return header;
	}
	
	public void getFacialFeatures(List<String> URLs, String img_path, String fileOut) throws IOException
	{
		FileWriter FW = new FileWriter(fileOut, true);
		for (String URL : URLs) {
			try {
				MBFImage query = ImageUtilities.readMBF(new URL(URL));
				FaceDetector<KEDetectedFace,FImage> fd = new FKEFaceDetector();
				List<KEDetectedFace> faces = fd.detectFaces( query.flatten());
				
				FW.write("{");
				FW.write(URL + ", ");
				for ( KEDetectedFace kd : faces)
				{
					
					FW.write(new Float (kd.getConfidence()).toString());
					Rectangle rect = kd.getBounds();
					FW.write(new Float(rect.x) .toString());
					FW.write(new Float(rect.y) .toString());
					FW.write(new Float(rect.height) .toString());
					FW.write(new Float(rect.width) .toString());
					FImage  fmg = kd.getFacePatch();
					FW.write("{");
					FW.write(fmg.height);
					FW.write(fmg.width);
					for (int i : fmg.toPackedARGBPixels())
					{
						FW.write(i + ", ");
					}
					FW.write("}");
					FW.write("{");
					for (FacialKeypoint k : kd.getKeypoints()) {
						
						// System.out.println("in sift");
						Float x = new Float(k.position.x);
						Float y = new Float(k.position.y);
						FW.write(x.toString() + ", ");
						FW.write(y.toString() + ", ");
						FW.write(k.type.toString());
						
					}
					FW.write("}");
					
				}
				FW.write( " \"" + url2access.get(URL) + "\"}\n");
			} catch (MalformedURLException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		FW.close();
	}
	
	public void getSIFTFeatures(List<String> URLs, String img_path, String fileOut) throws IOException
	{
		FileWriter FW = new FileWriter(fileOut, true);
		for (String URL : URLs) {
			try {
				MBFImage query = ImageUtilities.readMBF(new URL(URL));
				DoGSIFTEngine engine = new DoGSIFTEngine();
				LocalFeatureList <Keypoint > queryKeypoints = engine.findFeatures(query.flatten());
				List<Keypoint>keys =  queryKeypoints.subList(0, queryKeypoints.size()-1);
				FW.write("{");
				FW.write(URL + ", ");
				for ( Keypoint k : keys)
				{
					//System.out.println("in sift");
					Float x = new Float(k.x);
					Float y = new Float(k.y);
					Float ori = new Float(k.ori);
					Float scale = new Float(k.scale);
					byte[] vec = k.ivec;
					FW.write(x.toString() + ", ");
					FW.write(y.toString()+ ", ");
					FW.write(ori.toString()+ ", ");
					FW.write(scale.toString()+ ", ");
					FW.write("{");
					for (byte b : vec)
					{
						FW.write(b + ", ");
					}
					FW.write("}");
				}
				FW.write(" \"" + url2access.get(URL) +"\"}\n");
			} catch (MalformedURLException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		FW.close();
	}
	
	public void getEdgeDirectionCoherenceFeatures(List<String> URLs, String img_path, String fileOut) throws IOException
	{
		FileWriter FW = new FileWriter(fileOut, true);
		for (String URL : URLs) {
			try {
				MBFImage query = ImageUtilities.readMBF(new URL(URL));
				EdgeDirectionCoherenceVector edch = new EdgeDirectionCoherenceVector();
				FImage fmg = query.flatten();
				edch.analyseImage(fmg);
				double [] dblarr = edch.getFeatureVector().values;
				FW.write("{");
				FW.write(URL + ", ");
				FW.write( new Double( edch.getCoherenceFactor()) .toString());
				for ( double d : dblarr)
				{
					//System.out.println("in sift");
					FW.write(new Double(d) . toString() + ", ");
				}
				FW.write("{");
				FW.write(fmg.height);
				FW.write(fmg.width);
				for (int i : fmg.toPackedARGBPixels())
				{
					FW.write(i + ", ");
				}
				FW.write("}");
				FW.write( fmg.toPackedARGBPixels().length + " \"" +url2access.get(URL) + "\"}\n");
			} catch (MalformedURLException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		FW.close();
	}
	
	
	public void getRGBRepresentation(List<String> URLs, String img_path, String fileOut) throws Exception {
		
		FileWriter FW = new FileWriter(fileOut);
		
		FW.write(header.toString() + "\n\n");
		
		for (String URL : URLs)
		{
			
			//RGB features
			int[] vector = this.getVector( URL );
			FW.write("{");
			FW.write(URL + ", ");
			for (int j = 0; j < vector.length; j ++){
				if(vector[j] != 0)
					FW.write(j + " " + vector[j] + ", ");
			}
			FW.write(vector.length + " \"" + url2access.get(URL) + "\"}\n");
		}
		FW.close();
	}
	
	public int[] getVector(String img_path){
		
		int[] vector = new int[nf];
		
		int[][] values = new int[CONSTANT.numFeatures][CONSTANT.numValues];	//[0][]=red  [1][]=green [2][]=blue etc
		values = ImageUtils.extractRGB(img_path);
		
		int k = 0;
		for(int i=0;i<CONSTANT.numFeatures;i++) {
			for(int j=0;j<CONSTANT.numValues;j++) {
				vector[k] = values[i][j];
				k++;
			}
		}
		
		return vector;
	}
	
	public static void main(String[] args){

		String path = "/home/rahul/images/src/data/";
		File csvFile = new File("/home/rahul/MT_out.csv");
		//String picture = "/Users/cornelia/Desktop/AD/data/train_images/explosion.jpg";
		//@SuppressWarnings("unused")
		//int[][] values = Utils.extractRGB(picture);
		
		
		String fileTrain = path + "train_images.txt";
		String fileOut = path + "train_images.arff";
		
		int nf = CONSTANT.numFeatures * CONSTANT.numValues;
		
		System.out.println("Loading train!");
		Sequences train = new Sequences();
		try {
			train.loadSequences(fileTrain);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Done loading train!");
		
		RGBInstances iRep = new RGBInstances();
		iRep.setNF(nf);
		iRep.setHeader();
		List<String> URLs = iRep.getStaticURL(csvFile);
		try {
			iRep.getRGBRepresentation(URLs, path, fileOut);
			iRep.getSIFTFeatures(URLs, path, path+ "SIFT.arff");
			iRep.getEdgeDirectionCoherenceFeatures(URLs, path, path+"EDGC.arff");
			iRep.getFacialFeatures(URLs, path, path+"Facial.arff");
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		System.out.println("done");
	}
}
