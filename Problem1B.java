import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Problem1B {

    public static class PairMapper extends Mapper<Object, Text, Text, IntWritable> {

        private Set<String> topWords = new HashSet<>();
        private int window;

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            window = conf.getInt("window", 1);

            URI[] files = context.getCacheFiles();
            if (files == null || files.length == 0) {
                throw new IOException("Top-50 cache file is missing. Add it via job.addCacheFile(...)");
            }

            File localizedFile = new File(new Path(files[0].getPath()).getName());
            if (!localizedFile.exists()) {
                localizedFile = new File(files[0].getPath());
            }
            if (!localizedFile.exists()) {
                throw new IOException("Could not locate localized top-50 cache file for URI: " + files[0]);
            }

            try (BufferedReader br = new BufferedReader(new FileReader(localizedFile))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    if (parts.length == 0 || parts[0].isEmpty()) {
                        continue;
                    }
                    topWords.add(parts[0].toLowerCase());
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] tokens = value.toString().toLowerCase().split("[^\\w']+");
            for (int i = 0; i < tokens.length; i++) {

                if (!topWords.contains(tokens[i])) continue;

                for (int j = Math.max(0, i - window);
                     j <= Math.min(tokens.length - 1, i + window);
                     j++) {

                    if (i == j) continue;

                    if (!topWords.contains(tokens[j])) continue;

                    String pair = "(" + tokens[i] + "," + tokens[j] + ")";
                    context.write(new Text(pair), new IntWritable(1));
                }
            }
        }
    }

    public static class PairReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable v : values) {
                sum += v.get();
            }

            context.write(key, new IntWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            System.err.println("Usage: Problem1B <input> <output> <top50file> <window>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        conf.setInt("window", Integer.parseInt(args[3]));

        Job job = Job.getInstance(conf, "Co-occurrence Pairs");

        job.setJarByClass(Problem1B.class);
        job.setMapperClass(PairMapper.class);
        job.setReducerClass(PairReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.addCacheFile(new Path(args[2]).toUri());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}