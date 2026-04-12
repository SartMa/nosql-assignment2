import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Problem1E_MapClassLevel_Stripes {

    public static class StripeMapper extends Mapper<Object, Text, Text, MapWritable> {

        private Set<String> topWords = new HashSet<>();
        private int window;
        // ← Class-level aggregation map
        private Map<String, Map<String, Integer>> classStripes = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            window = conf.getInt("window", 1);

            URI[] files = context.getCacheFiles();
            if (files == null || files.length == 0) {
                throw new IOException("Top-50 cache file is missing.");
            }

            File localizedFile = new File(new Path(files[0].getPath()).getName());
            if (!localizedFile.exists()) {
                localizedFile = new File(files[0].getPath());
            }

            if (localizedFile.exists()) {
                try (BufferedReader br = new BufferedReader(new FileReader(localizedFile))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] parts = line.trim().split("\\s+");
                        if (parts.length == 0 || parts[0].isEmpty()) continue;
                        topWords.add(parts[0].toLowerCase());
                    }
                }
            } else {
                Path cachePath = new Path(files[0]);
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(cachePath.getFileSystem(conf).open(cachePath)))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] parts = line.trim().split("\\s+");
                        if (parts.length == 0 || parts[0].isEmpty()) continue;
                        topWords.add(parts[0].toLowerCase());
                    }
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] tokens = value.toString().toLowerCase().split("[^\\w']+");

            for (int i = 0; i < tokens.length; i++) {

                String wi = tokens[i];
                if (wi.isEmpty() || !topWords.contains(wi)) continue;

                // Get or create stripe for wi
                Map<String, Integer> stripe = classStripes.computeIfAbsent(wi, k -> new HashMap<>());

                int start = Math.max(0, i - window);
                int end = Math.min(tokens.length - 1, i + window);

                for (int j = start; j <= end; j++) {
                    if (i == j) continue;
                    String wj = tokens[j];
                    if (wj.isEmpty() || !topWords.contains(wj)) continue;

                    // ← Accumulate across ALL lines
                    stripe.put(wj, stripe.getOrDefault(wj, 0) + 1);
                }
            }
        }

        @Override
        protected void cleanup(Context context)
                throws IOException, InterruptedException {
            // ← Emit once per word at the very end
            for (Map.Entry<String, Map<String, Integer>> entry : classStripes.entrySet()) {
                MapWritable stripeWritable = new MapWritable();
                for (Map.Entry<String, Integer> e : entry.getValue().entrySet()) {
                    stripeWritable.put(new Text(e.getKey()), new IntWritable(e.getValue()));
                }
                context.write(new Text(entry.getKey()), stripeWritable);
            }
        }
    }

    public static class StripeReducer extends Reducer<Text, MapWritable, Text, MapWritable> {

        @Override
        public void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {

            MapWritable result = new MapWritable();

            for (MapWritable stripe : values) {
                for (Map.Entry<Writable, Writable> entry : stripe.entrySet()) {
                    Text neighbor = (Text) entry.getKey();
                    IntWritable count = (IntWritable) entry.getValue();

                    if (result.containsKey(neighbor)) {
                        IntWritable existing = (IntWritable) result.get(neighbor);
                        result.put(neighbor, new IntWritable(existing.get() + count.get()));
                    } else {
                        result.put(neighbor, new IntWritable(count.get()));
                    }
                }
            }
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            System.err.println("Usage: Problem1E_MapClassLevel_Stripes <input> <output> <top50file> <window>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        conf.setInt("window", Integer.parseInt(args[3]));

        long startTime = System.currentTimeMillis();

        Job job = Job.getInstance(conf, "Stripes - Map Class Level");
        job.setJarByClass(Problem1E_MapClassLevel_Stripes.class);

        job.setMapperClass(StripeMapper.class);
        job.setReducerClass(StripeReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(MapWritable.class);

        job.addCacheFile(new Path(args[2]).toUri());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        boolean success = job.waitForCompletion(true);

        long endTime = System.currentTimeMillis();
        System.err.println("=== TOTAL RUNTIME (sec): " + (endTime - startTime) / 1000.0 + " ===");

        System.exit(success ? 0 : 1);
    }
}