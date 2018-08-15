package Problem1;

import java.io.*;
import java.util.*;

/**
 * Created by ljb on 2018/1/17.
 */
public class Parser {
    //1976815
    /*
    public Map<String,Set<String>> author;
    public Map<String,Set<String>> venues;
    public Map<String,Set<String>> ref;
    public Map<String,Set<String>> cit;
    public Map<String,Set<String>> date;
    */

    public Map<String,String> author;
    public Map<String,String> venues;
    public Map<String,String> ref;
    public Map<String,String> cit;
    public Map<String,String> date;
    public Map<String,String> table;

    /*
     * author-publication count
     * venues-publication count
     * publication-reference count/citations count
     * year-publication
     */
    Parser(){
        author = new HashMap<String,String>();
        venues = new HashMap<String,String>();
        ref = new HashMap<String,String>();
        cit = new HashMap<String,String>();
        date = new HashMap<String,String>();
        table = new HashMap<String,String>();
    }

    public void checkcount() throws IOException{
        File theNewFile = new File("AP_train.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        int count = 0 ;
        while ((line = br.readLine()) != null) {
            if (line.startsWith("#index")) {
                count++;
                System.out.println(count);
            }
        }
        br.close();
        fis.close();
    }
    /*
    public void start1() throws IOException {
        File theNewFile = new File("AP_train.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        int count = 0 ;
        while ((line = br.readLine()) != null) {
            if(line.startsWith("#index")) {
                String index = line.substring(7);
                count++;
                System.out.println(count);
                line = br.readLine();
                if(!line.startsWith("#* "))
                    throw new Error("Wrong");
                line = br.readLine();
                if(!line.startsWith("#@ "))
                    throw new Error("Wrong");
                else{
                    line = line.substring(3);
                    if(!line.equals("")) {
                        String[] names = line.split(";");
                        for(String n : names) {
                            if (author.containsKey(n))
                                author.get(n).add(index);
                            else {
                                Set<String> temp = new HashSet<>();
                                temp.add(index);
                                author.put(n,temp);
                            }
                        }
                    }
                }
                line = br.readLine();
                if(!line.startsWith("#t "))
                    throw new Error("Wrong");
                else{
                    line = line.substring(3);
                    if(!line.equals("")) {
                        if (date.containsKey(line))
                            date.get(line).add(index);
                        else {
                            Set<String> temp = new HashSet<>();
                            temp.add(index);
                            date.put(line,temp);
                        }
                    }
                }
                line = br.readLine();
                if(!line.startsWith("#c "))
                    throw new Error("Wrong");
                else{
                    line = line.substring(3);
                    if(!line.equals("")) {
                        if (venues.containsKey(line))
                            venues.get(line).add(index);
                        else {
                            Set<String> temp = new HashSet<>();
                            temp.add(index);
                            venues.put(line,temp);
                        }
                    }
                }
                while((line = br.readLine()).startsWith("#%")){
                    line = line.substring(3);
                    if(!line.equals("")) {
                        if (cit.containsKey(line))
                            cit.get(line).add(index);
                        else {
                            Set<String> temp = new HashSet<>();
                            temp.add(index);
                            cit.put(line,temp);
                        }
                        if (ref.containsKey(index))
                            ref.get(index).add(line);
                        else {
                            Set<String> temp = new HashSet<>();
                            temp.add(line);
                            ref.put(index,temp);
                        }
                    }
                }
            }
        }
        br.close();
        fis.close();
    }
    */
    public void start2() throws IOException {
        File theNewFile = new File("AP_train.txt");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        int count = 0 ;
        while ((line = br.readLine()) != null) {
            if(line.startsWith("#index")) {
                String index = line.substring(7);
                count++;
                System.out.println(count);
                line = br.readLine();
                if(!line.startsWith("#* "))
                    throw new Error("Wrong");
                else{
                    line = line.substring(3);
                    if((!line.equals("")) || !line.equals(" ")){
                        table.put(index,line);
                    }
                }
                line = br.readLine();
                if(!line.startsWith("#@ "))
                    throw new Error("Wrong");
                else{
                    line = line.substring(3);
                    if((!line.equals("")) || !line.equals(" ")){
                        String[] names = line.split(";");
                        for(String n : names) {
                            if (author.containsKey(n)) {
                                String v = author.get(n) + " " + index;
                                author.put(n,v);
                            } else
                                author.put(n,index);
                        }
                    }
                }
                line = br.readLine();
                if(!line.startsWith("#t "))
                    throw new Error("Wrong");
                else{
                    line = line.substring(3);
                    if((!line.equals("")) && !line.equals(" ")){
                        if (date.containsKey(line)){
                            String v = date.get(line) + " " + index;
                            date.put(line,v);
                        } else
                            date.put(line,index);
                    }
                }
                line = br.readLine();
                if(!line.startsWith("#c "))
                    throw new Error("Wrong");
                else{
                    line = line.substring(3);
                    if((!line.equals("")) && !line.equals(" ")){
                        if (venues.containsKey(line)){
                            String v = venues.get(line) + " " + index;
                            venues.put(line,v);
                        } else
                            venues.put(line,index);
                    }
                }
                while((line = br.readLine()).startsWith("#%")){
                    line = line.substring(3);
                    if((!line.equals("")) && !line.equals(" ")){
                        if (cit.containsKey(line)){
                            String v = cit.get(line) + " " + index;
                            cit.put(line,v);
                        } else
                            cit.put(line,index);
                        if (ref.containsKey(index)){
                            String v = ref.get(index) + " " + line;
                            ref.put(index,v);
                        } else
                            ref.put(index,line);
                    }
                }
            }
        }
        br.close();
        fis.close();
    }

    public void record() throws IOException {
        File dir = new File("P1/");
        dir.mkdirs();

        File theNewFile = new File("P1/author.txt");
        theNewFile.createNewFile();
        FileWriter fw = new FileWriter(theNewFile);
        for(String key:author.keySet())
            fw.write(key+"#"+author.get(key)+"\n");
        fw.close();

        theNewFile = new File("P1/venues.txt");
        theNewFile.createNewFile();
        fw = new FileWriter(theNewFile);
        for(String key:venues.keySet())
            fw.write(key+"#"+venues.get(key)+"\n");
        fw.close();

        theNewFile = new File("P1/ref.txt");
        theNewFile.createNewFile();
        fw = new FileWriter(theNewFile);
        for(String key:ref.keySet())
            fw.write(key+"#"+ref.get(key)+"\n");
        fw.close();

        theNewFile = new File("P1/cit.txt");
        theNewFile.createNewFile();
        fw = new FileWriter(theNewFile);
        for(String key:cit.keySet())
            fw.write(key+"#"+cit.get(key)+"\n");
        fw.close();

        theNewFile = new File("P1/date.txt");
        theNewFile.createNewFile();
        fw = new FileWriter(theNewFile);
        for(String key:date.keySet())
            fw.write(key+"#"+date.get(key)+"\n");
        fw.close();

        theNewFile = new File("P1/table.txt");
        theNewFile.createNewFile();
        fw = new FileWriter(theNewFile);
        for(String key:table.keySet())
            fw.write(key+"#"+table.get(key)+"\n");
        fw.close();
    }

}
