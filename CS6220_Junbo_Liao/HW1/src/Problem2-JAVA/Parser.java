package Problem2;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by ljb on 2018/1/21.
 */
public class Parser {
    List<String> components;

    Parser(){
        components = new ArrayList<String>();
    }

    public void start() throws IOException {
        File theNewFile = new File("kosarak.dat");
        FileInputStream fis = new FileInputStream(theNewFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        int maxId = 0;
        int count = 0;
        while ((line = br.readLine()) != null) {
            count++;
            System.out.println(count);
            String[] ids = line.split(" ");
            int[] idnums = new int[ids.length];
            for(int i=0;i<ids.length;i++){
                idnums[i] = Integer.parseInt(ids[i]);
            }
            Arrays.sort(idnums);
            maxId = Math.max(maxId,idnums[idnums.length-1]);
            String content = "{";
            for(int i = 0;i<idnums.length;i++) {
                if (i == 0) {
                    content = content + Integer.toString(idnums[i] - 1) + " 1";
                } else if (idnums[i] != idnums[i - 1]) {
                    content = content + "," + Integer.toString(idnums[i] - 1) + " 1";
                }
            }
            content = content + "}";
            components.add(content);
        }

        File dir = new File("P2/");
        dir.mkdirs();

        theNewFile = new File("P2/kosarak.arff");
        theNewFile.createNewFile();
        FileWriter fw = new FileWriter(theNewFile);
        fw.write("@RELATION test\n");
        for(int i=0;i<maxId;i++)
            fw.write("@ATTRIBUTE i"+Integer.toString(i+1)+" {0,1}\n");
        fw.write("@DATA\n");
        for(int i = 0;i<components.size();i++)
            fw.write(components.get(i)+"\n");
        fw.close();
    }
}
