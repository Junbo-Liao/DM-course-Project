package Problem1;

import java.io.*;
import java.lang.reflect.Array;
import java.util.*;

/**
 * Created by ljb on 2018/1/21.
 */
public class Answer {

    public static void main(String[] args) throws IOException {
        solution1A();
        solution1B();
        solution1CD();
        solution1E();
        solution1F();
        solution1GH();
        solution1I();
        solution1J();
    }

    public static void solution1A() throws IOException {
        File theNewFile = new File("AP_train.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        int count = 0 ;
        while ((line = br.readLine()) != null) {
            if(line.startsWith("#index"))
                count++;
        }
        br.close();
        fis.close();
        System.out.println("A:\nNumber of publications: " + count);

        theNewFile = new File("P1/author.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        line = null;
        count = 0 ;
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" "))
                count++;
        }
        br.close();
        fis.close();
        System.out.println("Number of authors: " + count);

        theNewFile = new File("P1/venues.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        line = null;
        count = 0 ;
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#"))
                count++;
        }
        br.close();
        fis.close();
        System.out.println("Number of venues: " + count);

        theNewFile = new File("P1/cit.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        line = null;
        count = 0 ;
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#"))
                count = count + line.split(" ").length;
        }
        br.close();
        fis.close();
        System.out.println("Number of citations: " + count);
    }

    public static void solution1B() throws IOException{
        File theNewFile = new File("P1/venues.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        int count = 0 ;
        while ((line = br.readLine()) != null) {
            if(!line.startsWith(" ")){
                String[] content = line.split("#");
                if(content[0].contains("Principles and Practice of Knowledge Discovery in Databases"))
                    count = count + content[1].split(" ").length;
            }
        }
        br.close();
        fis.close();
        System.out.println("B:\nNumber of venues' names associated: " + count);
    }

    public static void solution1CD() throws IOException{
        File theNewFile = new File("P1/author.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        Map<String,Integer> authorPub = new HashMap<String,Integer>();
        int sum = 0;
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = content[1].split(" ").length;
                authorPub.put(content[0],count);
                sum = sum + count;
            }
        }
        br.close();
        fis.close();
        ArrayList<Map.Entry<String, Integer>> mappingList = new ArrayList<Map.Entry<String, Integer>>(authorPub.entrySet());
        Collections.sort(mappingList, new Comparator<Map.Entry<String, Integer>>() {
            public int compare(Map.Entry<String, Integer> mapping1, Map.Entry<String, Integer> mapping2) {
                return mapping1.getValue().compareTo(mapping2.getValue());
            }
        });
        int size = mappingList.size();
        double min = (double)mappingList.get(0).getValue();
        double max = (double)mappingList.get(size-1).getValue();
        double mean = ((double)sum)/((double)size);
        double Q1 = (double)mappingList.get(size/4).getValue();
        double median = (double)mappingList.get(size/2).getValue();
        double Q3 = (double)mappingList.get(size*3/4).getValue();
        double stddev = 0;
        for(Map.Entry<String,Integer> e : mappingList)
            stddev = stddev + Math.pow((e.getValue()-mean),2);
        stddev = Math.sqrt(stddev/size);
        System.out.println("D:\nmin = " + min );
        System.out.println("max = " + max );
        System.out.println("mean = " + mean );
        System.out.println("Q1 = " + Q1 );
        System.out.println("median = " + median );
        System.out.println("Q3 = " + Q3 );
        System.out.println("stddev = " + stddev );

        int pub0_100 = 0;
        int pub100_200 = 0;
        int pub200_300 = 0;
        int pub300_400 = 0;
        int pub400_500 = 0;
        int pub500_600 = 0;
        int pub600_700 = 0;
        int pub700_800 = 0;
        int pub800_900 = 0;
        int pub900_1000 = 0;
        int pub1000Up = 0;
        for(Map.Entry<String,Integer> e : mappingList){
            int num = e.getValue();
            if(0<num && num<=100)
                pub0_100++;
            else if(num<=200)
                pub100_200++;
            else if(num<=300)
                pub200_300++;
            else if(num<=400)
                pub300_400++;
            else if(num<=500)
                pub400_500++;
            else if(num<=600)
                pub500_600++;
            else if(num<=700)
                pub600_700++;
            else if(num<=800)
                pub700_800++;
            else if(num<=900)
                pub800_900++;
            else if(num<=1000)
                pub900_1000++;
            else if(num>1000)
                pub1000Up++;
        }
        System.out.print("        int pub0_100 = "+ pub0_100 + "\n" +
                            "        int pub100_200 = "+ pub100_200 + "\n" +
                            "        int pub200_300 = "+ pub200_300 + "\n" +
                            "        int pub300_400 = "+ pub300_400 + "\n" +
                            "        int pub400_500 = "+ pub400_500 + "\n" +
                            "        int pub500_600 = "+ pub500_600 + "\n" +
                            "        int pub600_700 = "+ pub600_700 + "\n" +
                            "        int pub700_800 = "+ pub700_800 + "\n" +
                            "        int pub800_900 = "+ pub800_900 + "\n" +
                            "        int pub900_1000 = "+ pub900_1000 + "\n" +
                            "        int pub1000Up = "+ pub1000Up + "\n");
    }

    public static void solution1E() throws IOException{
        File theNewFile = new File("P1/venues.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        Map<String,Integer> venuesPub = new HashMap<String,Integer>();
        int sum = 0;
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = content[1].split(" ").length;
                venuesPub.put(content[0],count);
                sum = sum + count;
            }
        }
        br.close();
        fis.close();
        ArrayList<Map.Entry<String, Integer>> mappingList = new ArrayList<Map.Entry<String, Integer>>(venuesPub.entrySet());
        Collections.sort(mappingList, new Comparator<Map.Entry<String, Integer>>() {
            public int compare(Map.Entry<String, Integer> mapping1, Map.Entry<String, Integer> mapping2) {
                return mapping1.getValue().compareTo(mapping2.getValue());
            }
        });
        int size = mappingList.size();
        double min = (double)mappingList.get(0).getValue();
        double max = (double)mappingList.get(size-1).getValue();
        double mean = ((double)sum)/((double)size);
        double Q1 = (double)mappingList.get(size/4).getValue();
        double median = (double)mappingList.get(size/2).getValue();
        double Q3 = (double)mappingList.get(size*3/4).getValue();
        double stddev = 0;
        for(Map.Entry<String,Integer> e : mappingList)
            stddev = stddev + Math.pow((e.getValue()-mean),2);
        stddev = Math.sqrt(stddev/size);
        System.out.println("E:\nmin = " + min );
        System.out.println("max = " + max );
        System.out.println("mean = " + mean );
        System.out.println("Q1 = " + Q1 );
        System.out.println("median = " + median );
        System.out.println("Q3 = " + Q3 );
        System.out.println("stddev = " + stddev );
        System.out.println("Most Publications venue: " + mappingList.get(size-1).getKey() );

        int pub0_100 = 0;
        int pub100_200 = 0;
        int pub200_300 = 0;
        int pub300_400 = 0;
        int pub400_500 = 0;
        int pub500_600 = 0;
        int pub600_700 = 0;
        int pub700_800 = 0;
        int pub800_900 = 0;
        int pub900_1000 = 0;
        int pub1000Up = 0;
        for(Map.Entry<String,Integer> e : mappingList){
            int num = e.getValue();
            if(0<num && num<=1000)
                pub0_100++;
            else if(num<=2000)
                pub100_200++;
            else if(num<=3000)
                pub200_300++;
            else if(num<=4000)
                pub300_400++;
            else if(num<=5000)
                pub400_500++;
            else if(num<=6000)
                pub500_600++;
            else if(num<=7000)
                pub600_700++;
            else if(num<=8000)
                pub700_800++;
            else if(num<=9000)
                pub800_900++;
            else if(num<=10000)
                pub900_1000++;
            else if(num>10000)
                pub1000Up++;
        }
        System.out.print("        int pub0_100 = "+ pub0_100 + "\n" +
                "        int pub100_200 = "+ pub100_200 + "\n" +
                "        int pub200_300 = "+ pub200_300 + "\n" +
                "        int pub300_400 = "+ pub300_400 + "\n" +
                "        int pub400_500 = "+ pub400_500 + "\n" +
                "        int pub500_600 = "+ pub500_600 + "\n" +
                "        int pub600_700 = "+ pub600_700 + "\n" +
                "        int pub700_800 = "+ pub700_800 + "\n" +
                "        int pub800_900 = "+ pub800_900 + "\n" +
                "        int pub900_1000 = "+ pub900_1000 + "\n" +
                "        int pub1000Up = "+ pub1000Up + "\n");
    }

    public static void solution1F() throws IOException{
        File theNewFile = new File("P1/cit.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        Map<String,Integer> Pubcit = new HashMap<String,Integer>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = content[1].split(" ").length;
                Pubcit.put(content[0],count);
            }
        }
        br.close();
        fis.close();
        theNewFile = new File("P1/ref.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        Map<String,Integer> Pubref = new HashMap<String,Integer>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = content[1].split(" ").length;
                Pubref.put(content[0],count);
            }
        }
        br.close();
        fis.close();
        theNewFile = new File("P1/table.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        Map<String,String> Pub = new HashMap<String,String>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                if(content.length>1)
                    Pub.put(content[0],content[1]);
            }
        }
        br.close();
        fis.close();

        ArrayList<Map.Entry<String, Integer>> mappingLista = new ArrayList<Map.Entry<String, Integer>>(Pubcit.entrySet());
        Collections.sort(mappingLista, new Comparator<Map.Entry<String, Integer>>() {
            public int compare(Map.Entry<String, Integer> mapping1, Map.Entry<String, Integer> mapping2) {
                return mapping1.getValue().compareTo(mapping2.getValue());
            }
        });
        System.out.println("F:\nThe publication with the largest number of citations:");
        int size = mappingLista.size();
        System.out.println(Pub.get(mappingLista.get(size-1).getKey()) + " " + mappingLista.get(size-1).getValue());
        //408396
        ArrayList<Map.Entry<String, Integer>> mappingListb = new ArrayList<Map.Entry<String, Integer>>(Pubref.entrySet());
        Collections.sort(mappingListb, new Comparator<Map.Entry<String, Integer>>() {
            public int compare(Map.Entry<String, Integer> mapping1, Map.Entry<String, Integer> mapping2) {
                return mapping1.getValue().compareTo(mapping2.getValue());
            }
        });
        System.out.println("The publication with the largest number of references: ");
        size = mappingListb.size();
        System.out.println(Pub.get(mappingListb.get(size-1).getKey()) + " " + mappingListb.get(size-1).getValue());
        //719353

        int pub0_100 = 0;
        int pub100_200 = 0;
        int pub200_300 = 0;
        int pub300_400 = 0;
        int pub400_500 = 0;
        int pub500_600 = 0;
        int pub600_700 = 0;
        int pub700_800 = 0;
        for(Map.Entry<String,Integer> e : mappingLista){
            int num = e.getValue();
            if(0<num && num<=1000)
                pub0_100++;
            else if(num<=2000)
                pub100_200++;
            else if(num<=3000)
                pub200_300++;
            else if(num<=4000)
                pub300_400++;
            else if(num<=5000)
                pub400_500++;
            else if(num<=6000)
                pub500_600++;
            else if(num<=7000)
                pub600_700++;
            else if(num>7000)
                pub700_800++;
        }
        System.out.print("        int pub0_100 = "+ pub0_100 + "\n" +
                "        int pub100_200 = "+ pub100_200 + "\n" +
                "        int pub200_300 = "+ pub200_300 + "\n" +
                "        int pub300_400 = "+ pub300_400 + "\n" +
                "        int pub400_500 = "+ pub400_500 + "\n" +
                "        int pub500_600 = "+ pub500_600 + "\n" +
                "        int pub600_700 = "+ pub600_700 + "\n" +
                "        int pub700_800 = "+ pub700_800 + "\n");

        pub0_100 = 0;
        pub100_200 = 0;
        pub200_300 = 0;
        pub300_400 = 0;
        pub400_500 = 0;
        pub500_600 = 0;
        pub600_700 = 0;
        pub700_800 = 0;
        for(Map.Entry<String,Integer> e : mappingListb){
            int num = e.getValue();
            if(0<num && num<=100)
                pub0_100++;
            else if(num<=200)
                pub100_200++;
            else if(num<=300)
                pub200_300++;
            else if(num<=400)
                pub300_400++;
            else if(num<=500)
                pub400_500++;
            else if(num<=600)
                pub500_600++;
            else if(num<=700)
                pub600_700++;
            else if(num>700)
                pub700_800++;
        }
        System.out.print("        int pub0_100 = "+ pub0_100 + "\n" +
                "        int pub100_200 = "+ pub100_200 + "\n" +
                "        int pub200_300 = "+ pub200_300 + "\n" +
                "        int pub300_400 = "+ pub300_400 + "\n" +
                "        int pub400_500 = "+ pub400_500 + "\n" +
                "        int pub500_600 = "+ pub500_600 + "\n" +
                "        int pub600_700 = "+ pub600_700 + "\n" +
                "        int pub700_800 = "+ pub700_800 + "\n");
    }

    public static void solution1GH() throws IOException{
        File theNewFile = new File("P1/cit.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        Map<String,Integer> Pubcit = new HashMap<String,Integer>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = content[1].split(" ").length;
                Pubcit.put(content[0],count);
            }
        }
        br.close();
        fis.close();

        theNewFile = new File("P1/venues.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        line = null;
        Map<String,Double> venuesCit = new HashMap<String,Double>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = 0;
                String[] pubs = content[1].split(" ");
                for(String pub:pubs)
                    if(Pubcit.containsKey(pub))
                        count = count + Pubcit.get(pub);
                double impact = ((double)count)/((double)pubs.length);
                venuesCit.put(content[0],impact);
            }
        }
        br.close();
        fis.close();

        ArrayList<Map.Entry<String, Double>> mappingList = new ArrayList<Map.Entry<String, Double>>(venuesCit.entrySet());
        Collections.sort(mappingList, new Comparator<Map.Entry<String, Double>>() {
            public int compare(Map.Entry<String, Double> mapping1, Map.Entry<String, Double> mapping2) {
                return mapping1.getValue().compareTo(mapping2.getValue());
            }
        });
        int size = mappingList.size();
        System.out.println("G&H:\nThe venue with the highest apparent impact factor:");
        System.out.println(mappingList.get(size-1).getKey() + " " + mappingList.get(size-1).getValue());

        int pub0_100 = 0;
        int pub100_200 = 0;
        int pub200_300 = 0;
        int pub300_400 = 0;
        int pub400_500 = 0;
        int pub500_600 = 0;
        int pub600_700 = 0;
        int pub700_800 = 0;
        for(Map.Entry<String,Double> e : mappingList){
            Double num = e.getValue();
            if(0<num && num<=1000)
                pub0_100++;
            else if(num<=2000)
                pub100_200++;
            else if(num<=3000)
                pub200_300++;
            else if(num<=4000)
                pub300_400++;
            else if(num<=5000)
                pub400_500++;
            else if(num<=6000)
                pub500_600++;
            else if(num<=7000)
                pub600_700++;
            else if(num>7000)
                pub700_800++;
        }
        System.out.print("        int pub0_100 = "+ pub0_100 + "\n" +
                "        int pub100_200 = "+ pub100_200 + "\n" +
                "        int pub200_300 = "+ pub200_300 + "\n" +
                "        int pub300_400 = "+ pub300_400 + "\n" +
                "        int pub400_500 = "+ pub400_500 + "\n" +
                "        int pub500_600 = "+ pub500_600 + "\n" +
                "        int pub600_700 = "+ pub600_700 + "\n" +
                "        int pub700_800 = "+ pub700_800 + "\n");
    }

    public static void solution1I() throws IOException{
        File theNewFile = new File("P1/cit.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        Map<String,Integer> Pubcit = new HashMap<String,Integer>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = content[1].split(" ").length;
                Pubcit.put(content[0],count);
            }
        }
        br.close();
        fis.close();

        theNewFile = new File("P1/venues.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        line = null;
        Map<String,Double> venuesCit = new HashMap<String,Double>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = 0;
                String[] pubs = content[1].split(" ");
                if(pubs.length >= 10) {
                    for (String pub : pubs)
                        if (Pubcit.containsKey(pub))
                            count = count + Pubcit.get(pub);
                    double impact = ((double)count)/((double)pubs.length);
                    venuesCit.put(content[0],impact);
                }
            }
        }
        br.close();
        fis.close();

        ArrayList<Map.Entry<String, Double>> mappingList = new ArrayList<Map.Entry<String, Double>>(venuesCit.entrySet());
        Collections.sort(mappingList, new Comparator<Map.Entry<String, Double>>() {
            public int compare(Map.Entry<String, Double> mapping1, Map.Entry<String, Double> mapping2) {
                return mapping1.getValue().compareTo(mapping2.getValue());
            }
        });
        int size = mappingList.size();
        System.out.println("I:\nThe venue with the highest apparent impact factor:");
        System.out.println(mappingList.get(size-1).getKey() + " " + mappingList.get(size-1).getValue());

        String maxVenue = mappingList.get(size-1).getKey();
        double mean = mappingList.get(size-1).getValue();
        theNewFile = new File("P1/venues.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        line = null;
        Map<String,Integer> maxVenuePub = new HashMap<String, Integer>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                if(content[0].equals(maxVenue)){
                    String[] pubs = content[1].split(" ");
                    for (String pub : pubs)
                        if (Pubcit.containsKey(pub))
                            maxVenuePub.put(pub,Pubcit.get(pub));
                    break;
                }
            }
        }
        br.close();
        fis.close();

        int pub0_100 = 0;
        int pub100_200 = 0;
        int pub200_300 = 0;
        int pub300_400 = 0;
        int pub400_500 = 0;
        int pub500_600 = 0;
        int pub600_700 = 0;
        int pub700_800 = 0;
        for(Map.Entry<String,Double> e : mappingList){
            Double num = e.getValue();
            if(0<num && num<=30)
                pub0_100++;
            else if(num<=60)
                pub100_200++;
            else if(num<=90)
                pub200_300++;
            else if(num<=120)
                pub300_400++;
            else if(num<=150)
                pub400_500++;
            else if(num<=180)
                pub500_600++;
            else if(num<=210)
                pub600_700++;
            else if(num>210)
                pub700_800++;
        }
        System.out.print("        int pub0_100 = "+ pub0_100 + "\n" +
                "        int pub100_200 = "+ pub100_200 + "\n" +
                "        int pub200_300 = "+ pub200_300 + "\n" +
                "        int pub300_400 = "+ pub300_400 + "\n" +
                "        int pub400_500 = "+ pub400_500 + "\n" +
                "        int pub500_600 = "+ pub500_600 + "\n" +
                "        int pub600_700 = "+ pub600_700 + "\n" +
                "        int pub700_800 = "+ pub700_800 + "\n");

        ArrayList<Map.Entry<String, Integer>> mappingList2 = new ArrayList<Map.Entry<String, Integer>>(maxVenuePub.entrySet());
        Collections.sort(mappingList2, new Comparator<Map.Entry<String, Integer>>() {
            public int compare(Map.Entry<String, Integer> mapping1, Map.Entry<String, Integer> mapping2) {
                return mapping1.getValue().compareTo(mapping2.getValue());
            }
        });

        size = mappingList2.size();
        System.out.println("The Publications List of Max Impact Venue:");
        for(int i=0;i<size;i++)
            System.out.println("Index: "+mappingList2.get(i).getKey() + "  Citations: "
                    + mappingList2.get(i).getValue());
        System.out.println("mean: " + mean + "  median: " + mappingList2.get(size/2).getValue());
    }
    //histogram
    public static void solution1J() throws IOException{
        File theNewFile = new File("P1/cit.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        Map<String,Integer> Pubcit = new HashMap<String,Integer>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = content[1].split(" ").length;
                Pubcit.put(content[0],count);
            }
        }
        br.close();
        fis.close();

        theNewFile = new File("P1/ref.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        Map<String,Integer> Pubref = new HashMap<String,Integer>();
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int count = content[1].split(" ").length;
                Pubref.put(content[0],count);
            }
        }
        br.close();
        fis.close();

        theNewFile = new File("P1/date.txt");
        fis = new FileInputStream(theNewFile);
        br = new BufferedReader(new InputStreamReader(fis));
        Map<String,Double> pubYearCit = new HashMap<String, Double>();
        Map<String,Double> pubYearRef = new HashMap<String, Double>();
        int startYear = 9999;
        int endYear = 0;
        while ((line = br.readLine()) != null) {
            if(!line.startsWith("#") && !line.startsWith(" ")){
                String[] content = line.split("#");
                int year = Integer.parseInt(content[0]);
                startYear = Math.min(year,startYear);
                endYear = Math.max(year,endYear);
                String[] Pubs = content[1].split(" ");
                int count = Pubs.length;
                int cits = 0;
                int refs = 0;
                for(String pub : Pubs){
                    if(Pubcit.containsKey(pub))
                        cits = cits + Pubcit.get(pub);
                    if(Pubref.containsKey(pub))
                        refs = refs + Pubref.get(pub);
                }
                //System.out.println(content[0] + " " + count + " " + cits + " " + refs);
                double res1 = ((double)cits)/((double)count);
                double res2 = ((double)refs)/((double)count);
                pubYearCit.put(content[0],res1);
                pubYearRef.put(content[0],res2);
            }
        }
        br.close();
        fis.close();
        System.out.println("J:\nCitations:");
        for(int i = startYear;i<=endYear;i++){
            String y = Integer.toString(i);
            if(pubYearCit.containsKey(y))
                System.out.println("Year: " + i + "  Average Number of Citations: "
                        + Double.toString(pubYearCit.get(y)));
        }
        System.out.println("References:");
        for(int i = startYear;i<=endYear;i++) {
            String y = Integer.toString(i);
            if (pubYearRef.containsKey(y))
                System.out.println("Year: " + i + "  Average Number of References: "
                        + Double.toString(pubYearRef.get(y)));
        }
    }
}
