__kernel void increment(__global int* in){
    int i= get_global_id(0);
    in[i]++;
}
